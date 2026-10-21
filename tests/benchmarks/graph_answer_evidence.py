'''Prepare and validate frozen graph and passage evidence for controlled answer comparisons.'''

import json
import tempfile
from pathlib import Path

from osintgpt import Project, index_project
from osintgpt.graph import build_graph, discover_entities, query_graph, saved_graph_status
from osintgpt.graph.answer_evaluation import answer_input, pack_context
from osintgpt.graph.export_snapshot import full_graph_package
from osintgpt.graph.recipes import extraction_recipe, fingerprint
from osintgpt.ingestion import Corpus
from osintgpt.prompts import prompt
from osintgpt.search import hybrid_search
from osintgpt.vector_store import store_for
from graph_benchmark import model_roles


def controls(fixture, budget, output_tokens=1024):
    return {'fixture_sha256': fingerprint(fixture), 'context_tokens': budget, 'output_tokens': output_tokens, 'encoding': 'cl100k_base',
            'answer_prompt_sha256': fingerprint(prompt('answer', question='', passages=[])),
            'retrieval_policy': {'version': 1, 'hybrid_top_k': 6, 'path_max_depth': 4,
                                 'identity_selection': 'unique_fixture_name_and_source',
                                 'graph_first': True, 'alternate_arm_order': True}}


def validate_bundle(bundle, fixture, budget, output_tokens=1024):
    if not isinstance(bundle, dict) or bundle.get('format_version') != 1:
        raise ValueError('A version-1 evidence bundle is required for reuse.')
    if bundle.get('sha256') != fingerprint({key: value for key, value in bundle.items() if key != 'sha256'}):
        raise ValueError('Saved evidence bundle fingerprint does not match its contents.')
    if (not {'controls', 'graph_build', 'rows', 'models', 'graph_snapshot', 'extraction_recipe'} <= bundle.keys()
            or not isinstance(bundle['graph_build'], dict) or not isinstance(bundle['rows'], list)
            or not isinstance(bundle['models'], dict) or not {'extraction', 'embedding'} <= bundle['models'].keys()):
        raise ValueError('Saved evidence bundle is missing required graph or model metadata.')
    if bundle['controls'] != controls(fixture, budget, output_tokens):
        raise ValueError('Evidence reuse requires the same fixture, input/output budgets, answer prompt and retrieval policy.')
    if bundle['graph_build'].get('status') not in ('complete', 'complete_with_gaps'):
        raise ValueError('Evidence reuse requires a finished graph build; failed or interrupted sources cannot be compared.')
    required = {'question_id', 'variant', 'question', 'context', 'system_input', 'expected_refs', 'graph_status'}
    if any(not isinstance(row, dict) or not required <= row.keys() for row in bundle['rows']):
        raise ValueError('Saved evidence rows are missing required inputs.')
    expected = {(q['name'], arm) for q in fixture['questions'] for arm in ('hybrid', 'graph_assisted')}
    actual = [(row['question_id'], row['variant']) for row in bundle['rows']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Saved evidence must cover both arms of every question exactly once.')
    return bundle


def selected_identity(project, fixture, value):
    '''Translate explicit fixture IDs using source anchors; never force namesake merges.'''
    anchors = [(item[side], item['ref']) for item in fixture['documents'] for side in ('source', 'target')
               if value is not None and item.get(side + '_id') == value]
    if not anchors:
        return value
    matches = set()
    for name, ref in anchors:
        found = discover_entities(project, name, limit=60, max_bytes=65536)
        if found.get('truncated'):
            return value
        matches.update(row['id'] for row in found.get('candidates', [])
                       if row['name'] == name and any(source['ref'] == ref for source in row['sources']))
    return next(iter(matches)) if len(matches) == 1 else value


def prepare(fixture, extractor, embedder, usage, budget, report, output_tokens=1024):
    if not fixture.get('questions') or len({q['name'] for q in fixture['questions']}) != len(fixture['questions']):
        raise ValueError('Benchmark questions need unique names and must not be empty.')
    rows = []
    with tempfile.TemporaryDirectory(prefix='osintgpt-answers-') as root:
        project = Project.create('Paired answer benchmark', home=root).with_settings(graph_enabled=True)
        for item in fixture['documents']:
            source = Path(item['ref'])
            if source.is_absolute() or len(source.parts) != 1 or '..' in source.parts or '\\' in str(source) or ':' in str(source):
                raise ValueError('Benchmark documents must use filenames inside the isolated project.')
            (project.paths.root / item['ref']).write_text(item['text'], encoding='utf8')
            Corpus.load(project.paths.sources).register(item['ref'])
        with usage.stage('embedding'):
            indexed = index_project(project, embedder)
        if indexed.failed:
            raise ValueError('Benchmark indexing failed; do not compare incomplete indexes.')
        try:
            with usage.stage('extraction'):
                built = build_graph(project, extractor)
        finally:
            state = saved_graph_status(project, check_freshness=False)
            report['graph_build'] = {key: state[key] for key in ('status', 'counts', 'usage')}
        if built.status not in ('complete', 'complete_with_gaps'):
            raise ValueError('Benchmark graph build is incomplete; answer comparison was not started.')
        snapshot = full_graph_package(project)
        with store_for(project) as store:
            for number, question in enumerate(fixture['questions']):
                with usage.stage('embedding'):
                    hits = hybrid_search(project, question['query'], embedder, top_k=6, terms=question['terms'], store=store)
                passages = [{'citation': hit.result.chunk.citation, 'text': hit.text} for hit in hits]
                graph = query_graph(project, selected_identity(project, fixture, question['entity']),
                                    selected_identity(project, fixture, question.get('target')),
                                    direction=question['direction'], depth=4 if question.get('target') else 1)
                claims = graph.get('path', graph.get('claims', []))
                additions = [{'citation': claim['citation'], 'text': json.dumps(claim, ensure_ascii=False)} for claim in claims]
                variants = [('hybrid', passages), ('graph_assisted', [*additions, *passages])]
                for variant, evidence in (reversed(variants) if number % 2 else variants):
                    context = pack_context(evidence, budget, question=question['query'])
                    rows.append({'question_id': question['name'], 'question': question['query'], 'variant': variant,
                                 'context': context, 'system_input': answer_input(question['query'], context),
                                 'expected_refs': question['expected'], 'graph_status': graph if variant == 'graph_assisted' else None})
    bundle = {'format_version': 1, 'controls': controls(fixture, budget, output_tokens), 'rows': rows, 'graph_snapshot': snapshot,
              'graph_build': report['graph_build'], 'models': model_roles(extractor=extractor, embedder=embedder),
              'extraction_recipe': extraction_recipe(extractor)}
    bundle = json.loads(json.dumps(bundle, ensure_ascii=False))
    bundle['sha256'] = fingerprint(bundle)
    return bundle
