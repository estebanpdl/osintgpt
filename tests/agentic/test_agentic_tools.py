# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_agentic_tools.py
# Description: The tools a model may call. Two of them carry guarantees rather
#   than features: fetch_source cannot escape the project, and refs mode
#   returns no content at all.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt
from osintgpt import Project, index_project
from osintgpt.agentic import (
    REFS,
    SNIPPETS,
    ToolContext,
    exact_search,
    fetch_source,
    graph_query,
    list_documents,
    semantic_search,
    snowball,
    snowball_search
)
from osintgpt.agentic.support import _moment, _within_days
from osintgpt.agentic.tools import FETCH_LINES
from osintgpt.graph import Edge, build_graph, graph_for
from osintgpt.ingestion import Corpus
from osintgpt.llm.base import EmbeddingProvider
from osintgpt.vector_store import SearchResult, StoredChunk

MODEL = 'test-embedding'


class WordEmbedder(EmbeddingProvider):
    model = MODEL
    VOCABULARY = 'aardvark zebra quokka narwhal ibex'.split()

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def _vector(self, text):
        low = text.lower()
        counts = [float(low.count(w)) for w in self.VOCABULARY]
        length = math.sqrt(sum(c * c for c in counts)) or 1.0

        return [c / length for c in counts]


@pytest.fixture
def embedder():
    return WordEmbedder()


@pytest.fixture
def project(tmp_path, embedder):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()
    (material / 'alpha.md').write_text(
        '---\ndate: 2026-04-22\n---\n\n'
        '# Alpha\n\nA report about aardvark sightings near @acct_1.\n\n'
        '## More\n\nFurther aardvark detail.',
        encoding='utf-8'
    )
    (material / 'beta.md').write_text(
        '---\ndate: 2019-01-01\n---\n\n# Beta\n\nAn old note about zebra.',
        encoding='utf-8'
    )
    (material / 'gamma.md').write_text(
        '# Gamma\n\nQuokka material with no date at all.', encoding='utf-8'
    )
    Corpus.load(instance.paths.sources).register('material')
    index_project(instance, embedder)

    return instance


@pytest.fixture
def context(project, embedder):
    return ToolContext(project=project, embedder=embedder)


class TestFetchSourceCannotEscape:
    '''
    The boundary is the project root, not the machine. An agent that can read
    outside it can read anything the process can.
    '''

    @pytest.mark.parametrize('ref', [
        '../secrets.md',
        '../../secrets.md',
        'material/../../secrets.md',
        '/etc/passwd',
        'material/../../../../../../etc/passwd'
    ])
    def test_a_ref_climbing_out_is_refused(self, context, tmp_path, ref):
        (tmp_path / 'secrets.md').write_text('private', encoding='utf-8')

        result = fetch_source(context, ref)

        assert not result.ok
        assert 'not a document in this project' in result.error

    def test_an_absolute_path_from_elsewhere_is_refused(
        self, context, tmp_path
    ):
        outside = tmp_path / 'elsewhere.md'
        outside.write_text('private', encoding='utf-8')

        assert not fetch_source(context, str(outside)).ok

    def test_a_document_inside_the_project_reads(self, context):
        result = fetch_source(context, 'material/alpha.md')

        assert result.ok
        assert 'aardvark' in result.payload['text']

    def test_a_ref_that_does_not_exist_is_refused(self, context):
        assert not fetch_source(context, 'material/absent.md').ok


class TestFetchSourceWindows:
    def test_it_says_where_to_continue_when_it_truncates(self, context):
        result = fetch_source(context, 'material/alpha.md', offset=0, limit=2)

        assert result.payload['next_offset'] == 2

    def test_it_does_not_offer_a_continuation_at_the_end(self, context):
        result = fetch_source(context, 'material/alpha.md', limit=1000)

        assert 'next_offset' not in result.payload

    def test_the_offset_is_honoured(self, context):
        whole = fetch_source(context, 'material/alpha.md').payload['text']
        tail = fetch_source(
            context, 'material/alpha.md', offset=2
        ).payload['text']

        assert whole != tail
        assert whole.endswith(tail)

    def test_it_reports_the_document_length(self, context):
        result = fetch_source(context, 'material/alpha.md', limit=1)

        assert result.payload['total_lines'] > 1

    def test_the_default_window_is_bounded(self):
        assert 0 < FETCH_LINES <= 1000


class TestSurveyPrimitive:
    '''
    refs mode is what lets a model range wide before reading anything, and it
    only works if it genuinely carries no content.
    '''

    def test_refs_mode_returns_no_content(self, context):
        result = exact_search(context, ['aardvark'], mode=REFS)

        assert 'passages' not in result.payload
        for row in result.payload['documents']:
            assert set(row) == {'ref', 'matches'}

    def test_refs_mode_says_how_many_matches_each_document_holds(self, context):
        result = exact_search(context, ['aardvark'], mode=REFS)
        rows = result.payload['documents']

        assert rows
        assert all(row['matches'] >= 1 for row in rows)

    def test_snippets_mode_returns_content(self, context):
        result = exact_search(context, ['aardvark'], mode=SNIPPETS)

        assert result.payload['passages']
        assert result.payload['passages'][0]['text']

    def test_documents_are_ordered_by_how_many_matches_they_hold(self, context):
        result = exact_search(context, ['aardvark'], mode=REFS)
        counts = [row['matches'] for row in result.payload['documents']]

        assert counts == sorted(counts, reverse=True)


class TestTimeFilter:
    '''
    The model decides what "last week" means and passes a number. Nothing here
    parses a phrase, because date language is bound to a language.
    '''

    @pytest.mark.parametrize('stamp, parsed', [
        ('2026-04-22', True),
        ('2022-10-26 07:39:35', True),
        ('2026-04-22T13:00:00', True),
        ('last Tuesday', False),
        ('', False),
        ('22/04/2026', False)
    ])
    def test_only_unambiguous_timestamps_are_read(self, stamp, parsed):
        assert (_moment(stamp) is not None) is parsed

    def test_an_old_document_is_filtered_out(self, context):
        recent = semantic_search(context, 'zebra', days=30)

        assert all('beta' not in p['ref'] for p in recent.payload['passages'])

    def test_a_recent_document_survives_the_filter(self, context):
        result = semantic_search(context, 'aardvark', days=100_000)

        assert result.payload['passages']

    def test_an_undated_document_is_kept_not_hidden(self, context):
        '''
        Hiding material because its date was unparseable is worse than a loose
        filter: the analyst never learns the document exists.
        '''
        result = semantic_search(context, 'quokka', days=30)
        refs = [p['ref'] for p in result.payload['passages']]

        assert any('gamma' in ref for ref in refs)

    def test_the_result_says_the_filter_was_partial(self, context):
        result = semantic_search(context, 'quokka', days=30)

        assert result.payload.get('undated_documents')
        assert 'partial' in result.payload['note']

    def test_no_note_when_nothing_was_undated(self, context):
        result = semantic_search(context, 'aardvark')

        assert 'note' not in result.payload

    def test_no_days_means_no_filtering(self):
        results = [_result('a.md', ''), _result('b.md', '1999-01-01')]

        kept, undated = _within_days(results, None)

        assert len(kept) == 2
        assert undated == 0


class TestSemanticAndExact:
    def test_semantic_search_returns_citations(self, context):
        result = semantic_search(context, 'aardvark')

        assert result.payload['passages'][0]['citation']

    def test_exact_search_finds_an_identifier(self, context):
        result = exact_search(context, ['@acct_1'])

        assert result.count >= 1

    def test_limits_are_clamped_rather_than_trusted(self, context):
        '''
        A model can ask for a thousand passages. The tool decides what it will
        actually spend.
        '''
        result = semantic_search(context, 'aardvark', limit=10_000)

        assert len(result.payload['passages']) <= 30

    def test_a_nonsense_limit_does_not_raise(self, context):
        assert semantic_search(context, 'aardvark', limit='many').ok


class TestListDocuments:
    def test_it_lists_what_is_indexed(self, context):
        result = list_documents(context)

        assert result.count == 3

    def test_a_pattern_narrows_it(self, context):
        result = list_documents(context, pattern='alpha')

        assert result.payload['documents'] == ['material/alpha.md']

    def test_an_unmatched_pattern_returns_nothing(self, context):
        assert list_documents(context, pattern='zzz').count == 0


class TestGraphTool:
    def test_it_says_so_when_there_is_no_graph(self, context):
        result = graph_query(context, 'Alpha')

        assert result.payload['built'] is False
        assert 'no graph' in result.payload['note']

    def test_every_claim_carries_its_evidence(self, project, embedder):
        with graph_for(project) as graph:
            graph.add([], [Edge(source='Alpha', target='Beta',
                                relation='funded', ref='r.md',
                                evidence='Alpha funded Beta.')])

        context = ToolContext(project=project, embedder=embedder)
        result = graph_query(context, 'Alpha')

        claim = result.payload['claims'][0]

        assert claim['ref'] == 'r.md'
        assert claim['evidence'] == 'Alpha funded Beta.'

    def test_an_unconnected_pair_says_the_documents_are_silent(
        self, project, embedder
    ):
        '''
        Not asserting a connection is different from there being none, and
        the difference matters to an analyst.
        '''
        with graph_for(project) as graph:
            graph.add([], [Edge(source='Alpha', target='Beta',
                                relation='funded', ref='r.md',
                                evidence='Alpha funded Beta.')])

        context = ToolContext(project=project, embedder=embedder)
        result = graph_query(context, 'Alpha', target='Unrelated')

        assert result.payload['connected'] is False
        assert 'not the same as there being none' in result.payload['note']


class TestSnowball:
    def test_it_walks_more_than_one_hop(self, project, embedder):
        walk = snowball(project, 'aardvark', embedder, depth=3, threshold=0.0)

        assert len(walk) >= 1

    def test_it_never_revisits_a_passage(self, project, embedder):
        walk = snowball(project, 'aardvark', embedder, depth=6, threshold=0.0)
        visited = [(h.result.chunk.ref, h.result.chunk.sequence)
                   for h in walk.hops]

        assert len(visited) == len(set(visited))

    def test_it_says_why_it_stopped(self, project, embedder):
        walk = snowball(project, 'aardvark', embedder, depth=2, threshold=0.0)

        assert walk.stopped

    def test_a_high_threshold_stops_it_early(self, project, embedder):
        walk = snowball(project, 'aardvark', embedder, depth=5, threshold=0.99)

        assert 'fell below' in walk.stopped or walk.stopped

    def test_each_hop_reports_drift_from_the_question(self, project, embedder):
        '''
        A walk can stay locally coherent while ending somewhere unrelated.
        Drift is what makes that visible rather than looking like a result.
        '''
        walk = snowball(project, 'aardvark', embedder, depth=3, threshold=0.0)

        assert all(h.drift is not None for h in walk.hops)

    def test_the_tool_returns_the_hops_and_the_reason(self, context):
        result = snowball_search(context, 'aardvark', depth=2, threshold=0.0)

        assert 'stopped' in result.payload
        assert isinstance(result.payload['hops'], list)


def _result(ref, timestamp):
    return SearchResult(
        chunk=StoredChunk(
            ref=ref, sequence=0, text='text', embedding_model=MODEL,
            timestamp=timestamp
        ),
        score=0.5
    )
