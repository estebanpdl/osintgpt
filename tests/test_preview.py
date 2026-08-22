# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_preview.py
# Description: The dry run — what a folder would become, without embedding any
#   of it, and what it says about the parts still awaiting a decision.
# =================================================================================

# import modules
import json
import pytest

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL

# import osintgpt ingestion
from osintgpt.ingestion import FieldMapping, dry_run, preview_file

MAPPING = FieldMapping(
    content=('content',), timestamp='captured_at', author='author',
    identity='record_id'
)


@pytest.fixture
def corpus(tmp_path):
    '''A folder shaped like real collected material.'''
    (tmp_path / 'notes').mkdir()
    (tmp_path / 'notes' / 'briefing.md').write_text(
        '# Briefing\n\n' + ('An assessed paragraph. ' * 60 + '\n\n') * 6,
        encoding='utf-8'
    )
    (tmp_path / 'notes' / 'page.html').write_text(
        '<html><body><p>Page content.</p></body></html>', encoding='utf-8'
    )

    rows = ['record_id,author,captured_at,content,count']
    for i in range(50):
        rows.append(
            f'r{i:03d},account_{i % 5},2026-03-{(i % 28) + 1:02d},'
            f'"Collected record {i} with enough text to be content.",{i}'
        )
    (tmp_path / 'records.csv').write_text('\n'.join(rows), encoding='utf-8')
    (tmp_path / 'archive.zip').write_bytes(b'not a document')

    return tmp_path


class TestEmbedsNothing:
    def test_no_provider_is_constructed(self, corpus, monkeypatch):
        '''
        The point of a dry run is that it can be run repeatedly for free. If it
        could reach a provider it would stop being free and stop being offline.
        '''
        import osintgpt.llm as llm

        def refuse(*args, **kwargs):
            raise AssertionError('the dry run built a provider')

        monkeypatch.setattr(llm, 'build_embedding_provider', refuse)
        monkeypatch.setattr(llm, 'build_generation_provider', refuse)

        assert dry_run(corpus, mappings={'records.csv': MAPPING}).chunks > 0


class TestTotals:
    def test_counts_documents_chunks_and_tokens(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})

        assert run.documents == 52
        assert run.chunks >= run.documents
        assert run.tokens > 0

    def test_one_row_is_one_document(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})
        csv_preview = next(
            f for f in run.readable if f.path.name == 'records.csv'
        )

        assert csv_preview.documents == 50

    def test_a_long_prose_file_chunks_into_several(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})
        markdown = next(
            f for f in run.readable if f.path.name == 'briefing.md'
        )

        assert markdown.documents == 1
        assert markdown.chunks > 1

    def test_estimates_a_cost_for_a_priced_model(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})

        assert run.estimated_cost > 0

    def test_an_unpriced_model_reports_unknown_not_zero(self, corpus):
        run = dry_run(
            corpus, mappings={'records.csv': MAPPING},
            embedding_model='embed-does-not-exist'
        )

        assert run.estimated_cost is None
        assert 'cost unknown' in run.summary

    def test_the_summary_leads_with_what_would_be_indexed(self, corpus):
        summary = dry_run(corpus, mappings={'records.csv': MAPPING}).summary

        assert summary.index('documents') < summary.index('$')


class TestUnconfigured:
    def test_a_structured_file_without_a_mapping_is_held_back(self, corpus):
        run = dry_run(corpus)

        assert [f.path.name for f in run.unconfigured] == ['records.csv']

    def test_it_is_not_counted_in_the_totals(self, corpus):
        unmapped = dry_run(corpus)
        mapped = dry_run(corpus, mappings={'records.csv': MAPPING})

        assert unmapped.documents < mapped.documents

    def test_its_fields_are_offered_for_a_decision(self, corpus):
        run = dry_run(corpus)
        held = run.unconfigured[0]

        assert set(held.fields) == {
            'record_id', 'author', 'captured_at', 'content', 'count'
        }

    def test_the_summary_says_a_decision_is_pending(self, corpus):
        assert 'need field mapping' in dry_run(corpus).summary

    def test_prose_never_needs_configuration(self, corpus):
        run = dry_run(corpus)

        assert all(
            f.path.suffix == '.csv' for f in run.unconfigured
        )


class TestProblemFiles:
    def test_an_unsupported_extension_is_listed_not_raised(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})

        assert [p.name for p in run.unsupported] == ['archive.zip']

    def test_one_unreadable_file_does_not_stop_the_rest(self, corpus):
        '''
        A corpus with one corrupt document should still yield a picture of
        everything else, or the preview is useless exactly when it is needed.
        '''
        (corpus / 'broken.json').write_text('{not valid json', encoding='utf-8')

        run = dry_run(corpus, mappings={
            'records.csv': MAPPING,
            'broken.json': FieldMapping(content=('body',))
        })

        assert len(run.failed) == 1
        assert run.failed[0].path.name == 'broken.json'
        assert run.documents == 52

    def test_a_failed_file_reports_why(self, corpus):
        (corpus / 'broken.json').write_text('{not valid json', encoding='utf-8')

        run = dry_run(corpus, mappings={
            'broken.json': FieldMapping(content=('body',))
        })

        assert run.failed[0].problem


class TestWalking:
    def test_it_descends_into_subfolders(self, corpus):
        run = dry_run(corpus, mappings={'records.csv': MAPPING})
        names = {f.path.name for f in run.readable}

        assert 'briefing.md' in names

    def test_tooling_directories_are_skipped(self, corpus):
        junk = corpus / '.venv' / 'lib'
        junk.mkdir(parents=True)
        (junk / 'vendored.md').write_text('Not corpus.', encoding='utf-8')

        run = dry_run(corpus, mappings={'records.csv': MAPPING})

        assert all('vendored' not in f.path.name for f in run.readable)

    def test_a_single_file_can_be_previewed(self, corpus):
        run = dry_run(corpus / 'notes' / 'briefing.md')

        assert len(run.readable) == 1


class TestTokenCounting:
    def test_counts_for_the_embedding_model(self, corpus):
        '''
        Encodings differ between models, so a count taken for the wrong one is
        a cost estimate for a run nobody is making. Plain ASCII tokenizes the
        same under both, so this needs text that actually separates them.
        '''
        path = corpus / 'mixed.md'
        path.write_text(
            'OSINT analysis — 分析 — análisis — تحليل\n\n' * 40,
            encoding='utf-8'
        )

        default = preview_file(path)
        other = preview_file(path, embedding_model='gpt-4o')

        assert default.tokens > 0
        assert default.tokens != other.tokens

    def test_the_default_model_is_the_library_default(self, corpus):
        run = dry_run(corpus)

        assert run.embedding_model == DEFAULT_EMBEDDING_MODEL


class TestChunkCeiling:
    def test_a_smaller_ceiling_produces_more_chunks(self, corpus):
        wide = dry_run(corpus, mappings={'records.csv': MAPPING})
        narrow = dry_run(
            corpus, mappings={'records.csv': MAPPING}, max_chars=200
        )

        assert narrow.chunks > wide.chunks
