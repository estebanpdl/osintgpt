'''Provider-free work estimates from the same snapshots, recipes, and windows as builds.'''

from dataclasses import asdict

from osintgpt.ingestion.snapshots import load_snapshot
from osintgpt.llm.generation_pacing import GenerationLimits

from . import extraction
from .checkpoints import Checkpoints
from .contract import EXTRACTION_SCHEMA
from .events import check_cancelled
from .recipes import extraction_recipe, fingerprint, compatible_recipe
from .reconciliation import inventory, refresh_changes, refresh_reason
from .model_metadata import selection_metadata
from .store import GraphStore
from .windowing import ExtractionOptions
from .output_policy import prepare_policy, policy_data
from .identity_tools import RESOLVING_SCHEMA, interactive_supported


def preflight_graph(project, generator, *, options=None, request_limits=None, rebuild=False, incremental=False):
    if rebuild and incremental:
        raise ValueError('incremental and rebuild cannot be used together')
    options, limits = options or ExtractionOptions(), request_limits or GenerationLimits()
    path = project.paths.root / 'graph.sqlite'
    with GraphStore(path, read_only=True) if path.is_file() else GraphStore(':memory:') as graph:
        if incremental and not graph.is_built:
            raise ValueError('The graph has not been built yet; run an initial build first.')
        return _estimate(project, generator, graph, options, limits, rebuild)


def _estimate(project, generator, graph, options, limits, rebuild):
    entries = [*graph.state.entries().values(), *Checkpoints(graph).jobs().values()]
    policy = prepare_policy(generator, entries)
    options = policy.options(options)
    corpus, targets, missing, removed, images = inventory(project, graph)
    recipe_data = extraction_recipe(generator, options)
    recipe = fingerprint(recipe_data)
    journal = Checkpoints(graph)
    completed, jobs = graph.state.entries(), journal.jobs()
    mappings = corpus.mappings(project.paths.root)
    hint = getattr(generator, 'structured_output_hint', None)
    if hint is None:
        hint = bool(getattr(generator, 'supports_structured_output', False))
    interactive = getattr(generator, 'interactive_lookup_hint', None)
    if interactive is None:
        interactive = interactive_supported(generator)
    schema = (RESOLVING_SCHEMA if interactive else EXTRACTION_SCHEMA) if hint else None
    from .request_context import RequestContext
    context = RequestContext(generator, options)
    totals = dict(documents=len(targets) + len(missing), documents_unchanged=0,
                  documents_to_extract=0, documents_unavailable=len(missing), documents_removed=len(removed),
                  images_skipped=images, records=0, records_unstarted=0, records_at_call_limit=0,
                  windows_total=0, windows_saved=0, windows_pending=0, responses_saved=0,
                  extraction_calls_without_retries=0, optional_correction_calls=0, optional_identity_calls=0, maximum_http_attempts=0,
                  documents_model_changed=0, checkpoints_incompatible=0, windows_incompatible=0, responses_incompatible=0)
    details = [{'ref': ref, 'action': 'unavailable'} for ref in missing]
    details.extend({'ref': ref, 'action': 'remove'} for ref in removed)
    for ref, path in targets.items():
        check_cancelled()
        try:
            snapshot = load_snapshot(path, mappings.get(path.resolve()), ref=ref, cache_dir=project.paths.extracts)
            reason = refresh_reason(completed.get(ref), snapshot, recipe, recipe_data)
            job = jobs.get(ref)
            if not rebuild and not reason and not job:
                totals['documents_unchanged'] += 1
                details.append({'ref': ref, 'action': 'unchanged'})
                continue
            compatible = not rebuild and job and job['source_version'] == snapshot.version and compatible_recipe(job['recipe_data'], recipe_data)
            changes = refresh_changes(completed.get(ref), snapshot, recipe_data)
            totals['documents_model_changed'] += int(changes['model_changed'])
            checkpoint = {'present': bool(job), 'compatible': bool(compatible), 'windows_saved': 0, 'responses_saved': 0}
            if job:
                saved_records = journal.records(ref)
                checkpoint.update(windows_saved=sum(row['saved'] for row in saved_records.values()),
                                  responses_saved=sum(row['responses'] for row in saved_records.values()),
                                  **refresh_changes(job, snapshot, recipe_data))
                if not compatible:
                    totals['checkpoints_incompatible'] += 1
                    totals['windows_incompatible'] += checkpoint['windows_saved']
                    totals['responses_incompatible'] += checkpoint['responses_saved']
            records = journal.records(ref) if compatible else {}
            detail = {'ref': ref, 'action': 'resume' if compatible else 'extract',
                      'reason': 'Explicit rebuild' if rebuild else reason,
                      'missing_pages': list(snapshot.missing_pages), 'records': [],
                      'published': bool(completed.get(ref)), 'changes': changes, 'checkpoint': checkpoint}
            for document in snapshot.documents:
                readable = any(not span.missing for span in document.spans)
                parts = extraction.plan_windows(document.ref, document.text, snapshot.version, options, schema,
                    extraction.WINDOW_CHARS, interactive=interactive, legacy=journal.recipe(ref) if compatible else None,
                    request_context=context, submission=interactive and getattr(generator, 'graph_submission_tool', False))[2] if readable and document.text.strip() else []
                saved = records.get(document.ref, {})
                total = saved.get('total', len(parts))
                pending = max(0, total - saved.get('saved', 0))
                remaining = None
                corrections = pending * 2 if options.repair_quotes else 0
                identity_calls = 0
                record = {'ref': document.ref, 'windows_total': total, 'windows_saved': saved.get('saved', 0),
                          'windows_pending': pending, 'responses_saved': saved.get('responses', 0),
                          'responses_invalid': saved.get('responses_invalid', 0),
                          'responses_for_local_validation': saved.get('responses', 0) - saved.get('responses_invalid', 0),
                          'recovery': 'Compatible legacy work; validate locally before reuse.' if compatible and journal.recipe(ref).get('version') == 4
                                      else 'Matching requests can replay saved responses; changed requests need new generation.',
                          'attempts_remaining': remaining}
                detail['records'].append(record)
                totals['records'] += 1
                totals['records_unstarted'] += int(not saved)
                totals['records_at_call_limit'] += int(pending > 0 and remaining == 0)
                for key in ('windows_total', 'windows_saved', 'windows_pending', 'responses_saved'):
                    totals[key] += record[key]
                totals['extraction_calls_without_retries'] += pending
                totals['optional_correction_calls'] += corrections
                totals['optional_identity_calls'] += identity_calls
                # Follow-up counts are data-dependent; no fixed attempt ceiling is promised.
            totals['documents_to_extract'] += 1
            details.append(detail)
        except (OSError, ValueError) as error:
            totals['documents_unavailable'] += 1
            details.append({'ref': ref, 'action': 'unavailable', 'problem': str(error)})
    attempts = None if totals['windows_pending'] else 0
    totals['maximum_http_attempts'] = attempts
    totals['optional_identity_calls'] = None if interactive and attempts is None else 0
    totals['optional_correction_calls'] = None if options.repair_quotes and attempts is None else 0
    identity = recipe_data['provider']
    from osintgpt.llm.registry import GENERATION_BACKENDS
    backend = GENERATION_BACKENDS.get(identity.get('provider'))
    billable = getattr(generator, 'billable', not (backend and backend.local))
    return {'project': project.slug, 'dry_run': True, 'graph_enabled': project.settings.graph_enabled,
            'provider': identity, 'graph_model': selection_metadata(generator, identity),
            'counts': totals, 'details': details, 'extraction_budget': asdict(options),
            'request_limits': {'rates': asdict(limits.rates), 'retry': asdict(limits.retry)},
            'input_estimator': 'utf8_bytes_conservative',
            'output_policy': policy_data(generator), 'output_policy_label': policy.label(),
            'identity_lookup': {'mode': 'interactive' if interactive else 'non_interactive',
                                'maximum_followup_turns_per_window': None if interactive else 0,
                                'note': 'Adapter support declared; model capability is validated by provider responses. '
                                        'Unsupported models fail explicitly without automatic paid fallback.' if interactive else
                                        'This adapter has no controlled structured tool turns; only supplied context and identifiers can be reused.'},
            'model_context_verified': policy.max_input_tokens is not None,
            'pacing_estimator': 'input_bytes_plus_model_output' if policy.max_tokens else 'twice_input_bytes',
            'budget_cost_estimate_usd': None if attempts is None and billable else 0.0,
            'assumptions': [
                'No provider requests. Source reading may populate the local text snapshot cache.',
                'Saved raw responses may reduce calls after validation; pending-window counts include them.',
                'Useful identity and correction work has no fixed turn/call ceiling. Request totals and future spend are unknown.',
                'Provider-default output has no application token ceiling; output usage and cost bounds are unknown.',
                'Pacing reserves input bytes plus the known model output allowance, otherwise twice input bytes; estimates do not limit generation.',
                'Unresolved model limits are checked at build time where available; preview windows may change.',
                'Recorded usage and known catalogue prices permit estimates for completed calls, not a guaranteed future spend.',
                'Unreadable sources and missing PDF text require attention; estimates cover readable material.',
                'Sources and extraction settings are checked again when the build runs.']}
