# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_store_contract.py
# Description: One suite, every backend. A seam nothing checks is a seam that
#   drifts, and "swapping backends is configuration" is only true if the
#   backends actually agree.
# =================================================================================

# import modules
import math
import pytest

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt vector store
from osintgpt.vector_store import (
    BaseVectorEngine,
    QdrantVectorStore,
    SQLiteVectorStore,
    StoredChunk
)

MODEL = 'text-embedding-3-small'
OTHER_MODEL = 'all-MiniLM-L6-v2'


def chunk(ref, sequence=0, text='some text', model=MODEL, **fields):
    return StoredChunk(
        ref=ref, sequence=sequence, text=text, embedding_model=model, **fields
    )


def unit(*values):
    length = math.sqrt(sum(v * v for v in values)) or 1.0

    return [v / length for v in values]


# Postgres has no in-process mode, so this backend is exercised only when a
# server is actually reachable. Set it to run the contract against pgvector:
#
#   docker run -d --rm --name osintgpt-pg -p 5433:5432 #     -e POSTGRES_PASSWORD=osintgpt pgvector/pgvector:pg17
#   OSINTGPT_TEST_POSTGRES_DSN=postgresql://postgres:osintgpt@localhost:5433/postgres
POSTGRES_DSN_VAR = 'OSINTGPT_TEST_POSTGRES_DSN'

# Qdrant runs in local mode by default. Point this at a server to run the same
# suite against the real thing:
#
#   docker run -d --rm --name osintgpt-qdrant -p 6333:6333 qdrant/qdrant
#   OSINTGPT_TEST_QDRANT_HOST=localhost
QDRANT_HOST_VAR = 'OSINTGPT_TEST_QDRANT_HOST'


@pytest.fixture(params=['sqlite', 'qdrant', 'postgres'])
def store(request, tmp_path):
    '''
    Every backend, one suite.

    Qdrant runs in its local mode rather than against a container, because a
    suite that needs Docker is a suite that does not run. Postgres has no
    equivalent, so it is skipped unless a server is configured.
    '''
    if request.param == 'postgres':
        yield from _postgres_store()

        return

    if request.param == 'sqlite':
        engine = SQLiteVectorStore(':memory:')
        yield engine
        engine.close()

        return

    import os
    import uuid
    import warnings

    qdrant_client = pytest.importorskip('qdrant_client')
    collection = f'contract_{uuid.uuid4().hex[:12]}'
    host = os.environ.get(QDRANT_HOST_VAR, '')

    if host:
        # A real server, when one is offered. Local mode is a reimplementation
        # of Qdrant rather than Qdrant, so it can agree with us about
        # something the server does differently.
        # The same timeout production uses: the point of running against a
        # server is to hit what an operator hits.
        from osintgpt.vector_store.connection import TIMEOUT_SECONDS

        client = qdrant_client.QdrantClient(
            host=host, port=6333, timeout=TIMEOUT_SECONDS
        )
        engine = QdrantVectorStore(
            Settings(qdrant_host=host, qdrant_port=6333),
            collection=collection,
            client=client
        )
        try:
            yield engine
        finally:
            if client.collection_exists(collection):
                client.delete_collection(collection)
            client.close()

        return

    client = qdrant_client.QdrantClient(':memory:')
    with warnings.catch_warnings():
        # Local mode warns that payload indexes do nothing. True, and beside
        # the point here: they matter on a server, which is where the data
        # would be.
        warnings.filterwarnings('ignore', message='.*local Qdrant.*')
        engine = QdrantVectorStore(
            Settings(), collection=collection, client=client
        )

        yield engine

    client.close()


def _postgres_store():
    '''
    A real Postgres, or a skip. Each test gets its own table so an
    interrupted run cannot leave rows that make the next one pass.
    '''
    import os
    import uuid

    dsn = os.environ.get(POSTGRES_DSN_VAR, '')
    if not dsn:
        pytest.skip(f'set {POSTGRES_DSN_VAR} to run the pgvector contract')

    pytest.importorskip('psycopg')
    pytest.importorskip('pgvector')

    from osintgpt.vector_store.pgvector_store import PgVectorStore

    engine = PgVectorStore(
        Settings(postgres_dsn=dsn),
        collection=f'contract_{uuid.uuid4().hex[:12]}'
    )
    try:
        yield engine
    finally:
        with engine.connection.cursor() as cursor:
            cursor.execute(f'DROP TABLE IF EXISTS {engine.table}')
        engine.connection.commit()
        engine.close()


class TestInterface:
    def test_it_is_a_vector_engine(self, store):
        assert isinstance(store, BaseVectorEngine)

    def test_an_empty_store_counts_nothing(self, store):
        assert store.count() == 0
        assert store.refs() == []
        assert store.models() == []

    def test_an_empty_store_searches_without_raising(self, store):
        '''
        A project that has never been indexed is an ordinary state, not an
        error, and must not depend on a collection existing.
        '''
        assert store.search(unit(1, 0), MODEL) == []


class TestUpsert:
    def test_chunks_and_vectors_must_match(self, store):
        with pytest.raises(ValueError, match='matched by position'):
            store.upsert('a.md', [chunk('a.md')], [unit(1, 0), unit(0, 1)])

    def test_it_stores_what_it_was_given(self, store):
        assert store.upsert('a.md', [chunk('a.md')], [unit(1, 0)]) == 1
        assert store.count() == 1

    def test_re_indexing_replaces_rather_than_appends(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', 0, 'first'), chunk('a.md', 1, 'second')],
            [unit(1, 0), unit(0, 1)]
        )
        store.upsert('a.md', [chunk('a.md', 0, 'rewritten')], [unit(1, 1)])

        assert store.count() == 1
        assert store.chunks_for('a.md')[0].text == 'rewritten'

    def test_a_document_that_shrank_keeps_no_orphans(self, store):
        '''
        Five chunks becoming three must not leave the two that went. Derived
        point ids alone would; the delete-then-insert is what prevents it.
        '''
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'part {i}') for i in range(5)],
            [unit(1, i) for i in range(5)]
        )
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'part {i}') for i in range(3)],
            [unit(1, i) for i in range(3)]
        )

        assert store.count() == 3

    def test_it_leaves_other_documents_alone(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md')], [unit(0, 1)])
        store.upsert('a.md', [chunk('a.md', text='new')], [unit(1, 1)])

        assert sorted(store.refs()) == ['a.md', 'b.md']

    def test_an_emptied_document_clears_its_chunks(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('a.md', [], [])

        assert store.count() == 0


class TestSearch:
    @pytest.fixture
    def populated(self, store):
        store.upsert('a.md', [chunk('a.md', 0, 'about alpha')], [unit(1, 0, 0)])
        store.upsert('b.md', [chunk('b.md', 0, 'about beta')], [unit(0, 1, 0)])
        store.upsert('c.md', [chunk('c.md', 0, 'about gamma')], [unit(0, 0, 1)])

        return store

    def test_the_closest_vector_comes_first(self, populated):
        assert populated.search(unit(1, 0.1, 0), MODEL)[0].ref == 'a.md'

    def test_scores_are_similarities(self, populated):
        results = populated.search(unit(1, 0, 0), MODEL)

        assert results[0].score == pytest.approx(1.0, abs=1e-5)

    def test_results_are_ordered_best_first(self, populated):
        scores = [r.score for r in populated.search(unit(1, 0.5, 0.2), MODEL)]

        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_the_results(self, populated):
        assert len(populated.search(unit(1, 0, 0), MODEL, top_k=2)) == 2

    def test_asking_for_more_than_exists_returns_what_exists(self, populated):
        assert len(populated.search(unit(1, 0, 0), MODEL, top_k=100)) == 3

    def test_it_can_be_restricted_to_documents(self, populated):
        results = populated.search(unit(1, 0, 0), MODEL, refs=['b.md', 'c.md'])

        assert {r.ref for r in results} == {'b.md', 'c.md'}

    def test_an_empty_restriction_returns_nothing(self, populated):
        assert populated.search(unit(1, 0, 0), MODEL, refs=[]) == []


class TestModelIsolation:
    @pytest.fixture
    def mixed(self, store):
        store.upsert('a.md', [chunk('a.md', model=MODEL)], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md', model=OTHER_MODEL)], [unit(1, 0)])

        return store

    def test_search_sees_only_its_own_model(self, mixed):
        assert [r.ref for r in mixed.search(unit(1, 0), MODEL)] == ['a.md']

    def test_an_unknown_model_finds_nothing(self, mixed):
        assert mixed.search(unit(1, 0), 'never-used') == []

    def test_both_models_are_named(self, mixed):
        assert mixed.models() == sorted([MODEL, OTHER_MODEL])

    def test_counts_can_be_per_model(self, mixed):
        assert mixed.count() == 2
        assert mixed.count(MODEL) == 1

    def test_refs_can_be_per_model(self, mixed):
        assert mixed.refs(MODEL) == ['a.md']

    def test_purging_reclaims_the_leftovers(self, mixed):
        assert mixed.purge_other_models(keep=MODEL) == 1
        assert mixed.models() == [MODEL]

    def test_purging_an_empty_store_removes_nothing(self, store):
        assert store.purge_other_models(keep=MODEL) == 0


class TestDelete:
    def test_it_forgets_a_document(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md')], [unit(0, 1)])

        assert store.delete(['a.md']) == 1
        assert store.refs() == ['b.md']

    def test_deleting_nothing_is_not_an_error(self, store):
        assert store.delete([]) == 0

    def test_deleting_an_absent_document_removes_nothing(self, store):
        store.upsert('a.md', [chunk('a.md')], [unit(1, 0)])

        assert store.delete(['never-stored.md']) == 0

    def test_it_counts_every_chunk_it_removed(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', i) for i in range(4)],
            [unit(1, i) for i in range(4)]
        )

        assert store.delete(['a.md']) == 4


class TestProvenance:
    def test_everything_a_citation_needs_survives(self, store):
        store.upsert(
            'report.md',
            [chunk(
                'report.md',
                path='Report › Findings',
                timestamp='2026-04-22',
                author='an analyst',
                metadata={'type': 'synthesis'}
            )],
            [unit(1, 0)]
        )

        found = store.search(unit(1, 0), MODEL)[0].chunk

        assert found.path == 'Report › Findings'
        assert found.timestamp == '2026-04-22'
        assert found.author == 'an analyst'
        assert found.metadata == {'type': 'synthesis'}

    def test_metadata_survives_non_ascii(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', metadata={'título': 'Análisis — 分析'})],
            [unit(1, 0)]
        )

        found = store.search(unit(1, 0), MODEL)[0].chunk

        assert found.metadata['título'] == 'Análisis — 分析'

    def test_chunks_come_back_in_reading_order(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'part {i}') for i in range(5)],
            [unit(1, i) for i in range(5)]
        )

        assert [c.text for c in store.chunks_for('a.md')] == [
            f'part {i}' for i in range(5)
        ]

    def test_a_citation_names_the_section_when_there_is_one(self):
        assert chunk('report.md', path='A › B').citation == 'report.md › A › B'


class TestRealisticVectors:
    def test_vectors_round_trip_at_a_realistic_size(self, store):
        '''1536 dimensions is what a current OpenAI embedding returns.'''
        vector = unit(*[float(i % 7) + 0.5 for i in range(1536)])
        store.upsert('a.md', [chunk('a.md')], [vector])

        assert store.search(vector, MODEL)[0].score == pytest.approx(
            1.0, abs=1e-4
        )


class TestMatchText:
    '''
    The exact leg's floor. Every backend matches substrings the same way, or
    swapping one silently changes what an analyst can find.
    '''

    @pytest.fixture
    def populated(self, store):
        store.upsert('a.md', [chunk('a.md', 0, 'contacted @acct_1 twice')],
                     [unit(1, 0)])
        store.upsert('b.md', [chunk('b.md', 0, 'Анализ НАРРАТИВОВ в сети')],
                     [unit(0, 1)])
        store.upsert('c.md', [chunk('c.md', 0, 'margin was 50% overall')],
                     [unit(1, 1)])

        return store

    def test_a_substring_inside_a_sentence_is_found(self, populated):
        assert [c.ref for c in populated.match_text('@acct_1')] == ['a.md']

    def test_matching_folds_case_beyond_ascii(self, populated):
        '''
        The place an English-only assumption gets reintroduced. A term in
        Cyrillic must match text stored in another casing, as it would in
        Latin script.
        '''
        assert [c.ref for c in populated.match_text('нарративов')] == ['b.md']

    def test_pattern_characters_are_literal(self, populated):
        assert [c.ref for c in populated.match_text('50%')] == ['c.md']

    def test_an_empty_term_matches_nothing(self, populated):
        assert populated.match_text('') == []

    def test_a_term_nothing_holds_matches_nothing(self, populated):
        assert populated.match_text('never-written') == []

    def test_an_empty_store_matches_nothing(self, store):
        assert store.match_text('anything') == []

    def test_it_can_be_restricted_to_a_model(self, store):
        store.upsert('a.md', [chunk('a.md', 0, 'shared text', model=MODEL)],
                     [unit(1, 0)])
        store.upsert('b.md',
                     [chunk('b.md', 0, 'shared text', model=OTHER_MODEL)],
                     [unit(1, 0)])

        found = store.match_text('shared text', embedding_model=MODEL)

        assert [c.ref for c in found] == ['a.md']

    def test_it_can_be_restricted_to_documents(self, populated):
        found = populated.match_text('a', refs=['b.md'])

        assert all(c.ref == 'b.md' for c in found)

    def test_an_empty_restriction_returns_nothing(self, populated):
        assert populated.match_text('a', refs=[]) == []

    def test_the_limit_is_respected(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'record {i} mentions nimbus') for i in range(10)],
            [unit(1, i) for i in range(10)]
        )

        assert len(store.match_text('nimbus', limit=3)) == 3

    def test_results_come_back_in_reading_order(self, store):
        store.upsert(
            'a.md',
            [chunk('a.md', i, f'part {i} nimbus') for i in range(5)],
            [unit(1, i) for i in range(5)]
        )
        found = store.match_text('nimbus')

        assert [c.sequence for c in found] == [0, 1, 2, 3, 4]
