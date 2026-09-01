'''Reading a scanned page once, and never paying for the same bytes twice.'''

import pytest

from osintgpt.ingestion.transcription import (
    cache_path,
    transcriber_for,
    transcriber_for_project
)
from osintgpt.projects import Project

PAGE = b'not really a png, but bytes are bytes'
OTHER = b'a different page entirely'


class Vision:
    '''A generation provider that records what it was asked to read.'''

    supports_vision = True
    model = 'test-vision'

    def __init__(self, reply='Transcribed page.'):
        self.reply = reply
        self.calls = []

    def describe_image(self, system, user, image, media_type='image/png'):
        self.calls.append({'user': user, 'image': image, 'media': media_type})

        return self.reply


@pytest.fixture
def cache(tmp_path):
    return tmp_path / 'extracts'


class TestTheCache:
    def test_the_same_bytes_are_read_once(self, cache):
        '''
        The whole reason the directory exists. Vision transcription is one
        generation call per page and the most expensive operation here.
        '''
        vision = Vision()
        transcribe = transcriber_for(vision, cache)

        first = transcribe(PAGE, 1)
        second = transcribe(PAGE, 1)

        assert first == second == 'Transcribed page.'
        assert len(vision.calls) == 1

    def test_a_different_page_is_read(self, cache):
        vision = Vision()
        transcribe = transcriber_for(vision, cache)

        transcribe(PAGE, 1)
        transcribe(OTHER, 2)

        assert len(vision.calls) == 2

    def test_the_cache_is_keyed_on_bytes_not_on_the_document(self, cache):
        '''
        So the same page re-registered under another name, or a document
        re-added after a move, costs nothing the second time.
        '''
        vision = Vision()
        transcribe = transcriber_for(vision, cache)

        transcribe(PAGE, 1)
        transcribe(PAGE, 97)

        assert len(vision.calls) == 1

    def test_it_survives_a_new_transcriber(self, cache):
        '''
        The case that actually costs money: `index --force` after changing the
        embedding model re-reads every document, and the transcription has
        nothing to do with which embedding model is in use.
        '''
        first = Vision()
        transcriber_for(first, cache)(PAGE, 1)

        second = Vision()
        text = transcriber_for(second, cache)(PAGE, 1)

        assert text == 'Transcribed page.'
        assert second.calls == []

    def test_the_page_number_reaches_the_prompt(self, cache):
        vision = Vision()
        transcriber_for(vision, cache)(PAGE, 7)

        assert 'page 7' in vision.calls[0]['user']

    def test_an_empty_transcription_is_not_cached(self, cache):
        '''
        A page that came back empty may have failed rather than been blank,
        and caching that would make one bad call permanent.
        '''
        vision = Vision(reply='   ')
        transcribe = transcriber_for(vision, cache)

        transcribe(PAGE, 1)
        transcribe(PAGE, 1)

        assert len(vision.calls) == 2

    def test_an_unwritable_cache_still_transcribes(self, cache, monkeypatch):
        '''
        The page was already read and paid for. Losing the cache write must
        not lose the transcription.
        '''
        def refuse(*_args, **_kwargs):
            raise OSError('read-only')

        monkeypatch.setattr('pathlib.Path.write_text', refuse)

        assert transcriber_for(Vision(), cache)(PAGE, 1) == 'Transcribed page.'

    def test_a_corrupt_cache_entry_is_a_miss(self, cache):
        vision = Vision()
        transcriber_for(vision, cache)(PAGE, 1)
        cache_path(cache, PAGE).write_bytes(b'\xff\xfe\x00 not utf-8')

        assert transcriber_for(vision, cache)(PAGE, 1) == 'Transcribed page.'
        assert len(vision.calls) == 2


class TestProjectTranscriber:
    def test_it_writes_into_the_project_extracts_directory(self, tmp_path):
        project = Project.create('Scan Case', home=tmp_path / 'home')
        vision = Vision()

        transcriber_for_project(project, lambda: vision)(PAGE, 1)

        assert cache_path(project.paths.extracts, PAGE).is_file()

    def test_the_generator_is_not_built_until_a_page_needs_it(self, tmp_path):
        '''
        A corpus of born-digital PDFs renders no page, and demanding a
        generation credential to index one would be charging for a model
        nothing ever asks anything.
        '''
        project = Project.create('Digital Case', home=tmp_path / 'home')
        built = []

        def build():
            built.append(1)

            return Vision()

        transcriber_for_project(project, build)

        assert built == []

    def test_the_generator_is_built_once_across_pages(self, tmp_path):
        project = Project.create('Many Pages', home=tmp_path / 'home')
        built = []

        def build():
            built.append(1)

            return Vision()

        transcribe = transcriber_for_project(project, build)
        transcribe(PAGE, 1)
        transcribe(OTHER, 2)

        assert len(built) == 1


class TestTheProvidersAcceptAnImage:
    def test_the_openai_shape_carries_a_data_uri(self):
        from osintgpt.llm.openai_compat import OpenAICompatGeneration

        provider = OpenAICompatGeneration(model='m', api_key='k')
        sent = {}

        class Completions:
            def create(self, **request):
                sent.update(request)

                class Reply:
                    choices = [type('C', (), {
                        'message': type('M', (), {'content': 'read'})()
                    })()]
                    usage = None

                return Reply()

        provider.client = type('Client', (), {
            'chat': type('Chat', (), {'completions': Completions()})()
        })()

        assert provider.describe_image('sys', 'read it', PAGE) == 'read'
        content = sent['messages'][1]['content']
        image = next(part for part in content if part['type'] == 'image_url')
        # A data URI rather than a hosted URL: uploading the page somewhere
        # fetchable would publish the document being analysed.
        assert image['image_url']['url'].startswith('data:image/png;base64,')

    def test_a_text_only_backend_says_so_rather_than_failing_obscurely(self):
        from osintgpt.llm.base import GenerationProvider

        class TextOnly(GenerationProvider):
            model = 'text-only'

            def generate(self, system, user):
                return ''

        with pytest.raises(NotImplementedError, match='image'):
            TextOnly().describe_image('sys', 'read it', PAGE)

    def test_vision_support_is_declared(self):
        from osintgpt.llm.base import GenerationProvider
        from osintgpt.llm.openai_compat import OpenAICompatGeneration

        assert OpenAICompatGeneration.supports_vision is True
        assert GenerationProvider.supports_vision is False
