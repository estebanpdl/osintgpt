# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_graph.py
# Description: The relational leg. Two guarantees matter more than the
#   traversal: an edge without its evidence is never stored, and the graph is
#   never built as a side effect of anything.
# =================================================================================

# import modules
import json
import pytest

# import osintgpt
from osintgpt import Project
from osintgpt.graph import (
    Edge,
    Entity,
    GraphStore,
    build_graph,
    extract_document,
    graph_for,
    merge_key,
    neighborhood,
    neighbors,
    path_between
)
from osintgpt.graph.extraction import WINDOW_CHARS
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import GenerationProvider


class ScriptedGenerator(GenerationProvider):
    '''Returns prepared replies in order, and counts its calls.'''

    model = 'test-generation'

    def __init__(self, *replies, error=None):
        self.replies = list(replies) or ['{"entities": [], "edges": []}']
        self.error = error
        self.calls = 0
        self.prompts = []

    def generate(self, system, user, **kwargs):
        self.calls += 1
        self.prompts.append(system)
        if self.error:
            raise self.error

        return self.replies[min(self.calls - 1, len(self.replies) - 1)]


def reply(entities=(), edges=()):
    return json.dumps({
        'entities': [
            {'name': n, 'type': t} for n, t in entities
        ],
        'edges': [
            {'source': s, 'relation': r, 'target': t, 'evidence': e}
            for s, r, t, e in edges
        ]
    })


@pytest.fixture
def store():
    with GraphStore(path=':memory:') as graph:
        yield graph


def edge(source, target, relation='linked to', ref='a.md', evidence='A sentence.'):
    return Edge(source=source, target=target, relation=relation,
                ref=ref, evidence=evidence)


class TestEvidenceIsRequired:
    '''
    An edge without the document and sentence asserting it is a model
    assertion, not a sourced claim. The difference is what makes the graph
    admissible in OSINT work at all.
    '''

    def test_an_edge_with_no_evidence_is_dropped(self):
        generator = ScriptedGenerator(json.dumps({
            'entities': [{'name': 'Alpha', 'type': 'org'}],
            'edges': [
                {'source': 'Alpha', 'relation': 'funded', 'target': 'Beta',
                 'evidence': 'Alpha funded Beta in March.'},
                {'source': 'Alpha', 'relation': 'knows', 'target': 'Gamma'}
            ]
        }))

        result = extract_document(generator, 'a.md', 'text')

        assert len(result.edges) == 1
        assert result.edges[0].target == 'Beta'

    def test_an_edge_with_a_blank_evidence_is_dropped(self):
        generator = ScriptedGenerator(
            reply(edges=[('A', 'r', 'B', '   ')])
        )

        assert extract_document(generator, 'a.md', 'text').edges == []

    def test_an_edge_missing_an_endpoint_is_dropped(self):
        generator = ScriptedGenerator(
            reply(edges=[('A', 'r', '', 'A sentence.')])
        )

        assert extract_document(generator, 'a.md', 'text').edges == []

    def test_a_stored_edge_carries_its_document_and_sentence(self, store):
        store.add([], [edge('Alpha', 'Beta', ref='report.md',
                            evidence='Alpha funded Beta in March.')])

        stored = store.edges()[0]

        assert stored.ref == 'report.md'
        assert stored.evidence == 'Alpha funded Beta in March.'

    def test_the_prompt_asks_for_the_sentence_and_forbids_inference(self):
        generator = ScriptedGenerator()
        extract_document(generator, 'a.md', 'text')
        system = generator.prompts[0].lower()

        assert 'evidence' in system
        assert 'do not infer' in system


class TestNeverBuiltAsASideEffect:
    '''
    One generation call per document. A graph that appears because someone
    asked a question is a bill nobody agreed to.
    '''

    @pytest.fixture
    def project(self, tmp_path):
        instance = Project.create('Case', home=tmp_path)
        material = instance.paths.root / 'material'
        material.mkdir()
        (material / 'a.md').write_text('# A\n\nAlpha funded Beta.',
                                       encoding='utf-8')
        Corpus.load(instance.paths.sources).register('material')

        return instance

    def test_a_project_with_the_graph_off_refuses(self, project):
        generator = ScriptedGenerator()

        report = build_graph(project, generator)

        assert report.refused
        assert generator.calls == 0

    def test_the_refusal_says_what_it_would_cost(self, project):
        report = build_graph(project, ScriptedGenerator())

        assert 'one generation call per document' in report.refused

    def test_graph_is_off_by_default(self, project):
        assert project.settings.graph_enabled is False

    def test_an_incremental_pass_before_the_first_build_refuses(self, project):
        '''
        The hard no-op: "keep it current" must never become "build it".
        '''
        enabled = project.with_settings(graph_enabled=True)
        enabled.save()
        generator = ScriptedGenerator()

        report = build_graph(enabled, generator, incremental=True)

        assert report.refused
        assert generator.calls == 0

    def test_an_explicit_build_runs(self, project):
        enabled = project.with_settings(graph_enabled=True)
        enabled.save()
        generator = ScriptedGenerator(
            reply([('Alpha', 'org'), ('Beta', 'org')],
                  [('Alpha', 'funded', 'Beta', 'Alpha funded Beta.')])
        )

        report = build_graph(enabled, generator)

        assert not report.refused
        assert generator.calls == 1
        assert report.edges == 1

    def test_the_report_says_how_many_calls_it_made(self, project):
        enabled = project.with_settings(graph_enabled=True)
        enabled.save()

        report = build_graph(enabled, ScriptedGenerator())

        assert report.calls == 1


class TestIncremental:
    @pytest.fixture
    def built(self, tmp_path):
        instance = Project.create('Case', home=tmp_path).with_settings(
            graph_enabled=True
        )
        instance.save()
        material = instance.paths.root / 'material'
        material.mkdir()
        (material / 'a.md').write_text('# A\n\nAlpha funded Beta.',
                                       encoding='utf-8')
        Corpus.load(instance.paths.sources).register('material')
        build_graph(instance, ScriptedGenerator(
            reply([('Alpha', 'org')],
                  [('Alpha', 'funded', 'Beta', 'Alpha funded Beta.')])
        ))

        return instance

    def test_an_unchanged_document_costs_no_call(self, built):
        generator = ScriptedGenerator()

        report = build_graph(built, generator, incremental=True)

        assert generator.calls == 0
        assert report.skipped == 1

    def test_a_new_document_is_read(self, built):
        (built.paths.root / 'material' / 'b.md').write_text(
            '# B\n\nGamma met Delta.', encoding='utf-8'
        )
        generator = ScriptedGenerator(
            reply([('Gamma', 'person')],
                  [('Gamma', 'met', 'Delta', 'Gamma met Delta.')])
        )

        report = build_graph(built, generator, incremental=True)

        assert generator.calls == 1
        assert report.skipped == 1

    def test_rebuilding_forgets_what_a_document_said_before(self, built):
        generator = ScriptedGenerator(
            reply([('Alpha', 'org')],
                  [('Alpha', 'no longer funded', 'Beta', 'Corrected.')])
        )

        build_graph(built, generator, rebuild=True)

        with graph_for(built) as graph:
            relations = [e.relation for e in graph.edges()]

        assert relations == ['no longer funded']


class TestStore:
    def test_an_identical_claim_is_stored_once(self, store):
        store.add([], [edge('A', 'B')])
        store.add([], [edge('A', 'B')])

        assert store.edge_count == 1

    def test_the_same_claim_from_two_documents_is_two_claims(self, store):
        '''
        Two sources asserting the same thing is corroboration, and collapsing
        them would hide it.
        '''
        store.add([], [edge('A', 'B', ref='one.md'),
                       edge('A', 'B', ref='two.md')])

        assert store.edge_count == 2

    def test_an_entity_keeps_its_first_spelling(self, store):
        store.add([Entity(key=merge_key('Alpha Corp'), name='Alpha Corp',
                          type='org', mentions=1)], [])
        store.add([Entity(key=merge_key('alpha corp'), name='alpha corp',
                          type='', mentions=1)], [])

        entities = store.entities()

        assert len(entities) == 1
        assert entities[0].name == 'Alpha Corp'
        assert entities[0].mentions == 2

    def test_forgetting_a_document_drops_its_claims_only(self, store):
        store.add([], [edge('A', 'B', ref='one.md'),
                       edge('C', 'D', ref='two.md')])

        assert store.forget(['one.md']) == 1
        assert [e.ref for e in store.edges()] == ['two.md']

    def test_an_empty_graph_is_not_built(self, store):
        assert store.is_built is False

    def test_it_becomes_built_once_something_is_stored(self, store):
        store.add([], [edge('A', 'B')])

        assert store.is_built is True


class TestMergeKeys:
    @pytest.mark.parametrize('a, b', [
        ('Alpha Corp', 'alpha corp'),
        ('Alpha  Corp', 'Alpha Corp'),
        (' Alpha Corp. ', 'Alpha Corp'),
        ('«Альфа»', 'Альфа'),
        ('ΑΛΦΑ', 'αλφα')
    ])
    def test_spellings_that_should_merge(self, a, b):
        assert merge_key(a) == merge_key(b)

    def test_accents_are_not_folded(self):
        '''
        `Bogota` and `Bogotá` are different strings, and merging them is a
        decision this layer is not entitled to make.
        '''
        assert merge_key('Bogotá') != merge_key('Bogota')

    def test_different_names_do_not_merge(self):
        assert merge_key('Alpha') != merge_key('Beta')


class TestTraversal:
    @pytest.fixture
    def populated(self, store):
        store.add([], [
            edge('Alpha', 'Beta', 'funded', 'one.md', 'Alpha funded Beta.'),
            edge('Beta', 'Gamma', 'owns', 'two.md', 'Beta owns Gamma.'),
            edge('Gamma', 'Delta', 'hired', 'three.md', 'Gamma hired Delta.'),
            edge('Epsilon', 'Zeta', 'met', 'four.md', 'Epsilon met Zeta.')
        ])

        return store

    def test_neighbors_returns_edges_at_either_end(self, populated):
        found = neighbors(populated, 'Beta')

        assert len(found) == 2

    def test_a_partial_name_matches(self, populated):
        store = populated
        store.add([], [edge('Project Nimbus', 'Cloud', 'is', 'x.md', 'S.')])

        assert neighbors(store, 'Nimbus')

    def test_every_hit_carries_its_evidence(self, populated):
        for hit in neighbors(populated, 'Beta'):
            assert hit.evidence
            assert hit.ref

    def test_a_path_is_the_shortest_chain(self, populated):
        path = path_between(populated, 'Alpha', 'Gamma')

        assert path.length == 2

    def test_a_path_names_the_documents_it_rests_on(self, populated):
        path = path_between(populated, 'Alpha', 'Gamma')

        assert path.refs == ['one.md', 'two.md']

    def test_an_unconnected_pair_returns_none(self, populated):
        '''
        None means the documents do not assert a connection, not that there
        is none.
        '''
        assert path_between(populated, 'Alpha', 'Epsilon') is None

    def test_depth_bounds_the_search(self, populated):
        assert path_between(populated, 'Alpha', 'Delta', max_depth=1) is None
        assert path_between(populated, 'Alpha', 'Delta', max_depth=3)

    def test_a_path_follows_edges_in_either_direction(self, populated):
        assert path_between(populated, 'Gamma', 'Alpha') is not None

    def test_neighborhood_reports_how_far_out_each_claim_was(self, populated):
        hits = neighborhood(populated, 'Alpha', depth=2)
        depths = {hit.depth for hit in hits}

        assert depths == {1, 2}

    def test_an_unknown_entity_finds_nothing(self, populated):
        assert neighbors(populated, 'Never Mentioned') == []

    def test_an_empty_name_finds_nothing(self, populated):
        assert neighbors(populated, '   ') == []
        assert path_between(populated, '', 'Beta') is None


class TestWholeDocuments:
    def test_a_short_document_is_one_call(self):
        generator = ScriptedGenerator()
        extract_document(generator, 'a.md', 'A short document.')

        assert generator.calls == 1

    def test_a_long_document_is_windowed(self):
        generator = ScriptedGenerator()
        extract_document(generator, 'a.md', 'word ' * 20_000)

        assert generator.calls > 1

    def test_later_windows_carry_the_names_already_found(self):
        '''
        Chunking breaks coreference. Carrying the entity list forward is what
        stops the same organization becoming two nodes across a boundary.
        '''
        generator = ScriptedGenerator(
            reply([('Alpha Corp', 'org')]),
            reply([('Alpha Corp', 'org')])
        )
        extract_document(generator, 'a.md', 'word ' * 20_000)

        assert 'Alpha Corp' in generator.prompts[1]

    def test_the_window_is_large_enough_that_most_documents_fit(self):
        assert WINDOW_CHARS >= 10_000

    def test_an_empty_document_costs_no_call(self):
        generator = ScriptedGenerator()
        extract_document(generator, 'a.md', '   ')

        assert generator.calls == 0


class TestFailures:
    def test_a_model_error_fails_one_document_not_the_pass(self):
        generator = ScriptedGenerator(error=RuntimeError('provider refused'))

        result = extract_document(generator, 'a.md', 'text')

        assert not result.ok
        assert 'provider refused' in result.problem

    def test_an_unparseable_reply_yields_nothing_rather_than_raising(self):
        result = extract_document(ScriptedGenerator('not json'), 'a.md', 'x')

        assert result.ok
        assert result.edges == []

    def test_a_reply_that_is_not_an_object_yields_nothing(self):
        result = extract_document(ScriptedGenerator('[1, 2]'), 'a.md', 'x')

        assert result.edges == []

    def test_json_wrapped_in_prose_is_still_read(self):
        generator = ScriptedGenerator(
            'Here you go:\n```json\n' +
            reply([('Alpha', 'org')]) + '\n```\nDone.'
        )

        assert extract_document(generator, 'a.md', 'x').entities
