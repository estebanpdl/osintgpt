'''Bounded extraction with persistent entity identities and source-only evidence gates.'''

from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional

from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import text_spans
from osintgpt.ingestion.snapshots import TextSnapshot
from osintgpt.prompts import prompt
from osintgpt.llm.generation_pacing import GenerationRequestError

from .accounting import ExtractionCounts
from .calling import ExtractionCalls, request_schema
from .contract import EXTRACTION_SCHEMA, REPAIR_SCHEMA, legacy_field_request
from .recipes import prior_relation_field
from .grounding import EvidenceScope, GroundingError, parse_response
from .identity import EntityRegistry
from .events import check_cancelled, emit_graph
from .records import Edge, Entity, RejectedClaim
from .validation import rejection, validate_edge
from .windowing import ExtractionOptions, Window, encode, units, windows
from .request_context import ProviderCapacityError, context_failure
from .output_policy import prepare_policy
from .completion import structured_generate
from .identity_tools import RESOLVING_SCHEMA, TOOLS, extract_with_lookup, interactive_supported

WINDOW_CHARS = 24_000
WINDOW_OVERLAP = 1_000


@dataclass(frozen=True)
class Extraction:
    ref: str
    entities: List[Entity] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    problem: str = ''
    accounting: ExtractionCounts = field(default_factory=ExtractionCounts)
    rejected: List[RejectedClaim] = field(default_factory=list)

    @property
    def ok(self):
        return not self.problem


def plan_windows(ref, text, version, options, schema, window_chars=WINDOW_CHARS, *, interactive=False, legacy=None,
                 request_context=None, submission=False):
    system = prompt('graph_extraction')
    envelope = {'document_ref': ref, 'source_version': version,
                'source': {'start': 0, 'end': len(text), 'text': ''}, 'known_entities': []}
    # A compatible saved job must retain its original source boundaries. The
    # longer field name only affects overhead; new provider requests still use
    # the current instructions and undergo the current capacity checks.
    sizing_system, sizing_schema, sizing_tools = (
        legacy_field_request(system, schema, TOOLS) if prior_relation_field(legacy)
        else (system, schema, TOOLS))
    room = options.context_budget
    if legacy and legacy.get('version') == 4:
        old = legacy['options']
        allowance = min(old['context_budget'], options.model_input_units or old['context_budget']) - 512
        room = allowance - units(sizing_system) - units(encode(envelope)) - old.get('context_units', 4096)
        if sizing_schema is not None:
            room -= units(encode(sizing_schema))
        if interactive:
            room -= units(encode(sizing_tools)) + min(8192, allowance // 3)
    elif options.model_input_units is not None:
        from .identity_session import SAVED, SUBMIT
        tools = ([*TOOLS, SAVED] + ([SUBMIT] if submission else [])) if interactive else None
        if prior_relation_field(legacy):
            _, _, tools = legacy_field_request(system, schema, tools)
        overhead = (request_context.measure(sizing_system, encode(envelope), None if submission else sizing_schema, tools)
                    if request_context else units(sizing_system) + units(encode(envelope)) + units(encode(sizing_schema)) + units(encode(tools)))
        room = min(room, options.model_input_units - overhead)
        if room < 64:
            raise ProviderCapacityError('Known provider input capacity cannot fit extraction instructions and a minimal source passage.')
    return system, envelope, windows(text, window_chars, room, WINDOW_OVERLAP)


def extract_document(generator, ref, text, window_chars=WINDOW_CHARS,
                     on_progress: Optional[Callable[[ExtractionCounts], None]] = None,
                     *, snapshot=None, document=None, known_entities=(), options=None, checkpoints=None, requests=None):
    if not (text or '').strip():
        return Extraction(ref)
    options = prepare_policy(generator, resolve=True).options(options or ExtractionOptions())
    document = document or Document(ref, text, spans=text_spans(text))
    if snapshot is None:
        import hashlib
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        snapshot = TextSnapshot(ref, digest, digest, (document,))
    interactive = interactive_supported(generator)
    contract = RESOLVING_SCHEMA if interactive else EXTRACTION_SCHEMA
    schema = request_schema(generator, contract)
    from .calling import generation_provider
    from .request_context import RequestContext
    provider = generation_provider(generator)
    context = RequestContext(provider, options)
    legacy = checkpoints.recipe(snapshot.ref) if checkpoints else None
    system, envelope, parts = plan_windows(ref, text, snapshot.version, options, schema, window_chars,
                                         interactive=interactive, legacy=legacy, request_context=context,
                                         submission=interactive and getattr(provider, 'graph_submission_tool', False))
    counts = ExtractionCounts(windows_total=len(parts))
    counts.interactive_lookup_unavailable = int(not interactive)
    checkpoint = checkpoints.record(snapshot.ref, ref, known_entities, len(parts)) if checkpoints else None

    if checkpoint:
        parts = checkpoint.plan(parts)
        counts.windows_total = len(parts)

    def progress(current):
        if checkpoint:
            checkpoint.progress(current)
        if on_progress:
            on_progress(current)

    calls = ExtractionCalls(generator, options, counts, progress, checkpoint, requests)
    registry = EntityRegistry(checkpoint.seed if checkpoint else known_entities)
    from .output_policy import identity
    registry.model = identity(generator)
    entities, edges, rejected = {}, [], []
    try:
        if checkpoint:
            from .checkpoints import entity_from, edge_from, rejection_from
            for position, saved in checkpoint.windows():
                if position != counts.windows_reused + 1 or position > len(parts):
                    raise ValueError('Graph checkpoints are not a contiguous window sequence; rebuild this graph.')
                found = {entity.key: entity for entity in map(entity_from, saved['entities'])}
                registry.commit(found.values())
                entities.update(found)
                edges.extend(map(edge_from, saved['edges']))
                rejected.extend(map(rejection_from, saved['rejected']))
                counts.windows_reused += 1
                counts.windows_completed += 1
        calls.notify()
        index = counts.windows_reused + 1
        while index <= len(parts):
            check_cancelled()
            window = parts[index - 1]
            calls.position = index
            registry.window = index
            emit_graph('window', ref=ref, window=index, total=len(parts))
            edge_start, audit_start = len(edges), len(rejected)
            context_size = (legacy['options'].get('context_units', 4096) if legacy and legacy.get('version') == 4
                            else min(options.context_budget, units(text[window.start:window.end])) if interactive
                            else options.context_budget)
            try:
                data, payload, scope, allowed = _window_request(calls, system, envelope, registry, document, snapshot,
                                                               window, context_size, interactive, contract, schema)
            except Exception as error:
                if not isinstance(error, ProviderCapacityError) and not context_failure(error):
                    raise
                length = window.end - window.start
                if length <= 1:
                    raise ProviderCapacityError('Provider capacity cannot fit the source instructions and minimal source text; saved work is preserved.') from error
                smaller = windows(text[window.start:window.end], max(1, length // 2),
                                  options.context_budget, WINDOW_OVERLAP)
                parts[index - 1:index] = [Window(window.start + part.start, window.start + part.end) for part in smaller]
                counts.windows_total = len(parts)
                if checkpoint:
                    checkpoint.replan(parts, index)
                calls.emit('partitioning_source', windows=len(parts))
                continue
            found, local = {}, {}
            failed_entities = []
            for row in data['entities']:
                try:
                    resolved, names = registry.resolve([row], scope, allowed)
                    if local.keys() & names.keys():
                        raise ValueError('Duplicate local entity ID in one response.')
                    found.update(resolved)
                    local.update(names)
                except (ValueError, TypeError) as error:
                    failed_entities.append((rejection(scope, index, {**row, 'kind': 'entity'}, error),
                                            isinstance(error, GroundingError)))
            entity_audit = _repair_entities(failed_entities, payload, scope, found, local, registry, allowed, calls)
            rejected.extend(entity_audit)
            counts.rejected_entities += sum(item.status == 'rejected' for item in entity_audit)
            for row, warning in registry.warnings:
                rejected.append(replace(rejection(scope, index, {**row, 'kind': 'entity_field'}, warning), status='warning'))
                counts.field_warnings += 1
            registry.warnings.clear()
            counts.identity_unresolved += sum(entity.provenance.get('resolution', {}).get('action') == 'unresolved'
                                              for entity in found.values())
            pending = []
            for row in data['edges']:
                counts.candidate_claims += 1
                try:
                    edges.append(validate_edge(row, scope, found, local, registry, allowed))
                except (ValueError, TypeError) as error:
                    item = rejection(scope, index, row, error)
                    pending.append((item, isinstance(error, GroundingError)))
            repaired, audit = _repair(pending, payload, scope, found, local, registry, allowed, calls)
            edges.extend(repaired)
            rejected.extend(audit)
            counts.rejected_claims += sum(item.status == 'rejected' for item in audit)
            entities.update(found)
            registry.commit(found.values())
            if checkpoint:
                checkpoint.save(index, found.values(), edges[edge_start:], rejected[audit_start:])
            counts.windows_completed += 1
            calls.notify()
            emit_graph('window_saved' if checkpoint else 'window_completed', ref=ref, window=index, total=len(parts))
            index += 1
    except GenerationRequestError:
        raise
    except Exception as error:
        return Extraction(ref, problem=str(error) or type(error).__name__, accounting=counts, rejected=rejected)
    finally:
        calls.notify()
    return Extraction(ref, list(entities.values()), _unique(edges), accounting=counts, rejected=rejected)


def _window_request(calls, system, envelope, registry, document, snapshot, window, context_size, interactive, contract, schema):
    known, ranges = registry.context(document, snapshot, window, context_size)
    payload = {**envelope, 'known_entities': known,
               'source': {'start': window.start, 'end': window.end, 'text': document.text[window.start:window.end]}}
    from .identity_session import SAVED, SUBMIT, extraction_system
    from .calling import generation_provider
    tools = [*TOOLS, SAVED] if interactive else None
    submission = interactive and getattr(generation_provider(calls.generator), 'graph_submission_tool', False)
    if submission:
        tools = [*tools, SUBMIT]
    replay = bool(known) and calls.has_saved_response(extraction_system(system, submission), encode(payload),
                                                     None if submission else schema,
                                                     tools=tools, history=[] if interactive else None)
    if known and not replay and not calls.context.fits(system, encode(payload), schema, tools):
        # Initial hints are optional; every identity remains available through lookup.
        payload['known_entities'], ranges = [], []
    scope = EvidenceScope(document, snapshot, [(window.start, window.end), *ranges])
    if interactive:
        reply, lookup = extract_with_lookup(calls, system, payload, registry)
        registry.candidate_evidence = lookup.evidence
        data = parse_response(reply, schema=schema)
    else:
        data = structured_generate(calls, system, payload, contract)
    allowed = {item['id'] for item in payload['known_entities']} | (lookup.allowed if interactive else set())
    return data, payload, scope, allowed


def _repair(pending, payload, scope, entities, local, registry, allowed, calls):
    if not pending:
        return [], []
    audit = [item for item, _ in pending]
    if not calls.options.repair_quotes:
        return [], audit
    data, problems = _correction_batches(pending, payload, calls, 'repair')
    audit = [replace(item, repair_problem=problems.get(i, '')) for i, item in enumerate(audit)]
    accepted, seen = [], set()
    eligible_indices = {i for i, (_, eligible) in enumerate(pending) if eligible and i not in problems}
    for correction in data['corrections']:
        index = correction.get('index')
        if type(index) is not int or index not in eligible_indices or index in seen:
            continue
        seen.add(index)
        row = {**audit[index].candidate, 'evidence': correction.get('evidence')}
        try:
            edge = validate_edge(row, scope, entities, local, registry, allowed)
        except (ValueError, TypeError) as error:
            audit[index] = replace(audit[index], repair_problem=str(error))
            continue
        accepted.append(edge)
        audit[index] = replace(audit[index], status='repaired', corrected_evidence=edge.evidence_quotes)
        calls.counts.repaired_claims += 1
    return accepted, audit


def _repair_entities(pending, payload, scope, found, local, registry, allowed, calls):
    if not pending:
        return []
    audit = [item for item, _ in pending]
    if not calls.options.repair_quotes:
        return audit
    data, problems = _correction_batches(pending, payload, calls, 'repair-entities')
    audit = [replace(item, repair_problem=problems.get(i, '')) for i, item in enumerate(audit)]
    eligible = {i for i, (_, supported) in enumerate(pending) if supported and i not in problems}
    for correction in data['corrections']:
        index = correction.get('index')
        if type(index) is not int or index not in eligible:
            continue
        eligible.remove(index)
        row = {**audit[index].candidate, 'evidence': correction.get('evidence')}
        row.pop('kind', None)
        try:
            resolved, names = registry.resolve([row], scope, allowed)
            if local.keys() & names.keys():
                raise ValueError('Duplicate local entity ID in correction.')
            found.update(resolved)
            local.update(names)
            audit[index] = replace(audit[index], status='repaired', corrected_evidence=tuple(row['evidence']))
            calls.counts.repaired_entities += 1
        except (ValueError, TypeError) as error:
            audit[index] = replace(audit[index], repair_problem=str(error))
    return audit


def _correction_batches(pending, payload, calls, purpose):
    system = prompt('graph_quote_repair')
    batches, batch, problems, corrections = [], [], {}, []
    for index, (item, eligible) in enumerate(pending):
        if not eligible:
            continue
        candidate = {'index': index, 'candidate': item.candidate}
        batch_purpose = purpose if not batches else f'{purpose}-batch-{len(batches)}'
        if batch and not calls.fits(system, encode({**payload, 'candidates': [*batch, candidate]}), REPAIR_SCHEMA,
                                    purpose=batch_purpose):
            batches.append(batch)
            batch = []
        batch_purpose = purpose if not batches else f'{purpose}-batch-{len(batches)}'
        if not batch and not calls.fits(system, encode({**payload, 'candidates': [candidate]}), REPAIR_SCHEMA,
                                        purpose=batch_purpose):
            problems[index] = 'Provider capacity cannot fit this correction and its source evidence; proposal retained for inspection.'
        else:
            batch.append(candidate)
    if batch:
        batches.append(batch)
    for number, batch in enumerate(batches):
        try:
            data = structured_generate(calls, system, {**payload, 'candidates': batch}, REPAIR_SCHEMA,
                                       purpose=purpose if number == 0 else f'{purpose}-batch-{number}', key='corrections')
            indices = {item['index'] for item in batch}
            corrections.extend(row for row in data['corrections'] if row.get('index') in indices)
        except GenerationRequestError:
            raise
        except Exception as error:
            problems.update({item['index']: str(error) for item in batch})
    return {'corrections': corrections}, problems


def _unique(edges):
    from dataclasses import asdict
    seen, kept = set(), []
    for edge in edges:
        marker = encode({key: value for key, value in asdict(edge).items() if key != 'provenance'})
        if marker not in seen:
            seen.add(marker)
            kept.append(edge)
    return kept


def _windows(text, window_chars):
    return [text[part.start:part.end] for part in windows(text, window_chars, window_chars * 4, WINDOW_OVERLAP)]
