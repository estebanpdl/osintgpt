# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_images.py
# Description: Standalone images, and the rule that matters more than indexing
#   them: a registered file is never dropped without saying so.
# =================================================================================

# import modules
import io
import math
import pytest

# import osintgpt
from osintgpt import Project, index_project
from osintgpt.ingestion import (
    IMAGE_SUFFIXES,
    Corpus,
    is_image,
    load_documents,
    marker_for
)
from osintgpt.ingestion.preview import dry_run
from osintgpt.llm.base import EmbeddingProvider
from osintgpt.vector_store import SQLiteVectorStore

MODEL = 'text-only-model'
MULTIMODAL = 'multimodal-model'


def unit(*values):
    length = math.sqrt(sum(v * v for v in values)) or 1.0

    return [v / length for v in values]


class TextOnlyEmbedder(EmbeddingProvider):
    '''The common case: a model that cannot see.'''

    model = MODEL
    supports_images = False

    def embed(self, texts):
        return [unit(1.0, float(len(text) % 7)) for text in texts]


class MultimodalEmbedder(EmbeddingProvider):
    '''One vector space for text and images alike.'''

    model = MULTIMODAL
    supports_images = True

    def __init__(self):
        self.images_seen = []

    def embed(self, texts):
        return [unit(1.0, float(len(text) % 7)) for text in texts]

    def embed_images(self, images):
        self.images_seen.extend(images)

        return [unit(float(len(data) % 5), 1.0) for data in images]


def write_png(path):
    '''A real PNG, so nothing downstream is fooled by a text file.'''
    pillow = pytest.importorskip('PIL.Image')
    buffer = io.BytesIO()
    pillow.new('RGB', (8, 8), color=(120, 20, 20)).save(buffer, format='PNG')
    path.write_bytes(buffer.getvalue())

    return path


@pytest.fixture
def project(tmp_path):
    instance = Project.create('Case', home=tmp_path)
    material = instance.paths.root / 'material'
    material.mkdir()

    (material / 'report.md').write_text(
        '# Report\n\nAssessed material.', encoding='utf-8'
    )
    write_png(material / 'evidence.png')

    Corpus.load(instance.paths.sources).register('material')

    return instance


class TestRecognition:
    @pytest.mark.parametrize('name', [
        'a.png', 'a.jpg', 'a.jpeg', 'a.webp', 'a.gif', 'a.bmp', 'a.tif'
    ])
    def test_raster_formats_are_images(self, name):
        assert is_image(name) is True

    def test_the_check_is_case_insensitive(self):
        assert is_image('PHOTO.PNG') is True

    def test_svg_is_not_an_image_here(self):
        '''
        It is markup. Treating it as a picture would embed a rendering of text
        the lexical leg could otherwise search directly.
        '''
        assert is_image('diagram.svg') is False
        assert '.svg' not in IMAGE_SUFFIXES

    def test_documents_are_not_images(self):
        for name in ['a.md', 'a.pdf', 'a.csv', 'a.docx']:
            assert is_image(name) is False

    def test_an_image_names_itself_by_filename(self, tmp_path):
        assert marker_for(tmp_path / 'evidence.png') == '[Image: evidence.png]'


class TestTheTextPathRefuses:
    def test_load_documents_says_why_rather_than_returning_nothing(
        self, tmp_path
    ):
        '''
        Returning an empty list would read as "this file is empty", which is a
        different thing from "this file is not text".
        '''
        path = write_png(tmp_path / 'evidence.png')

        with pytest.raises(ValueError, match='no text to load'):
            load_documents(path)

    def test_images_still_count_as_corpus(self, tmp_path):
        from osintgpt.ingestion import READABLE_SUFFIXES

        assert IMAGE_SUFFIXES <= READABLE_SUFFIXES

    def test_a_registered_folder_picks_them_up(self, project):
        files = Corpus.load(project.paths.sources).files(project.paths.root)

        assert any(f.name == 'evidence.png' for f in files)


class TestWithoutAMultimodalModel:
    '''
    The rule this sub-step exists for: never silently drop a file.
    '''

    def test_the_image_is_skipped_not_failed(self, project):
        report = index_project(project, TextOnlyEmbedder())

        assert len(report.skipped) == 1
        assert report.failed == []

    def test_the_notice_names_the_file_and_the_model(self, project):
        report = index_project(project, TextOnlyEmbedder())
        notice = report.notices[0]

        assert 'evidence.png' in notice
        assert MODEL in notice

    def test_the_notice_names_no_vendor(self, project):
        '''
        Several providers offer a multimodal embedding model, and which one an
        operator reaches for is their decision, not a recommendation osintgpt
        makes on their behalf.
        '''
        notice = index_project(project, TextOnlyEmbedder()).notices[0].lower()

        for vendor in ('openai', 'voyage', 'gemini', 'google', 'cohere',
                       'anthropic', 'clip'):
            assert vendor not in notice

    def test_the_summary_says_something_was_skipped(self, project):
        assert 'skipped' in index_project(project, TextOnlyEmbedder()).summary

    def test_the_documents_still_index(self, project):
        report = index_project(project, TextOnlyEmbedder())

        assert len(report.indexed) == 1
        assert report.indexed[0].ref.endswith('report.md')

    def test_nothing_is_stored_for_the_image(self, project):
        index_project(project, TextOnlyEmbedder())

        with SQLiteVectorStore(project.paths.store) as store:
            assert all('evidence' not in ref for ref in store.refs())

    def test_a_later_pass_reports_it_again(self, project):
        '''
        A skipped file must not be recorded as done, or an operator who
        switches to a multimodal model would never see it indexed.
        '''
        index_project(project, TextOnlyEmbedder())
        report = index_project(project, TextOnlyEmbedder())

        assert len(report.skipped) == 1


class TestWithAMultimodalModel:
    def test_the_image_is_embedded(self, project):
        embedder = MultimodalEmbedder()
        report = index_project(project, embedder)

        assert report.skipped == []
        assert len(embedder.images_seen) == 1

    def test_it_is_stored_as_one_chunk(self, project):
        index_project(project, MultimodalEmbedder())

        with SQLiteVectorStore(project.paths.store) as store:
            ref = next(r for r in store.refs() if 'evidence' in r)

            assert len(store.chunks_for(ref)) == 1

    def test_the_stored_text_is_a_marker_not_a_caption(self, project):
        '''
        Nothing was extracted. Inventing a description would put words in the
        index that no model produced.
        '''
        index_project(project, MultimodalEmbedder())

        with SQLiteVectorStore(project.paths.store) as store:
            ref = next(r for r in store.refs() if 'evidence' in r)

            assert store.chunks_for(ref)[0].text == '[Image: evidence.png]'

    def test_the_provider_receives_the_bytes_as_stored(self, project):
        embedder = MultimodalEmbedder()
        index_project(project, embedder)

        assert embedder.images_seen[0].startswith(b'\x89PNG')

    def test_it_switches_on_the_model_not_the_file(self, project):
        '''
        Same corpus, same files; only the embedding model differs.
        '''
        text_only = index_project(project, TextOnlyEmbedder())
        multimodal = index_project(
            project, MultimodalEmbedder(), force=True
        )

        assert len(text_only.skipped) == 1
        assert multimodal.skipped == []


class TestTheDefaultIsRefusal:
    def test_a_provider_says_no_unless_it_says_yes(self):
        assert EmbeddingProvider.supports_images is False

    def test_embedding_images_raises_by_default(self):
        with pytest.raises(NotImplementedError, match='text only'):
            TextOnlyEmbedder().embed_images([b'not an image'])

    def test_the_refusal_names_the_model(self):
        with pytest.raises(NotImplementedError, match=MODEL):
            TextOnlyEmbedder().embed_images([b''])


class TestDryRun:
    def test_images_are_counted(self, project):
        run = dry_run(project.paths.root / 'material')

        assert len(run.images) == 1

    def test_they_are_reported_before_anything_is_paid_for(self, project):
        run = dry_run(project.paths.root / 'material')

        assert 'multimodal' in run.summary

    def test_an_image_costs_no_tokens(self, project):
        '''There is nothing to tokenize, so it must not inflate an estimate.'''
        run = dry_run(project.paths.root / 'material')
        image = run.images[0]

        assert image.tokens == 0
        assert image.chunks == 1
