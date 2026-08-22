# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_deprecation.py
# Description: The pre-provider classes stay importable and keep working, warn
#   about it, and route their work through the provider layer.
# =================================================================================

# import modules
import ast
import pytest

# import submodules
from pathlib import Path

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt embeddings
from osintgpt.embeddings import OpenAIEmbeddingGenerator

# import osintgpt llm
from osintgpt.llm import EmbeddingProvider, GenerationProvider

# import osintgpt llms
from osintgpt.llms import OpenAIGPT

from conftest import FAKE_KEY

PACKAGE = Path(__file__).resolve().parent.parent / 'osintgpt'


@pytest.fixture
def settings(tmp_path):
    return Settings(
        openai_api_key=FAKE_KEY,
        openai_gpt_model='gpt-4o',
        sql_db_file_path=str(tmp_path / 'log.db')
    )


class TestStillImportable:
    def test_both_classes_still_exist(self):
        '''The package is on PyPI; someone's script imports these.'''
        assert OpenAIGPT is not None
        assert OpenAIEmbeddingGenerator is not None

    def test_the_generator_still_embeds(self, settings, stub_client):
        generator = OpenAIEmbeddingGenerator(settings)
        generator.client = stub_client
        generator.load_text(['a', 'b'])

        assert len(generator.calculate_embeddings()) == 2

    def test_the_chat_class_still_completes(self, settings, stub_client):
        gpt = OpenAIGPT(settings)
        gpt.client = stub_client

        assert gpt.get_model_completion('a question', verbose=False) == (
            'STUB REPLY'
        )


class TestWarns:
    def test_the_generator_names_its_replacement(self, settings):
        with pytest.warns(DeprecationWarning) as caught:
            OpenAIEmbeddingGenerator(settings)

        message = str(caught[0].message)

        assert 'build_embedding_provider' in message
        assert '1.0' in message

    def test_the_chat_class_names_its_replacement(self, settings):
        with pytest.warns(DeprecationWarning) as caught:
            OpenAIGPT(settings)

        message = str(caught[0].message)

        assert 'build_generation_provider' in message
        assert '1.0' in message

    def test_the_warning_points_at_the_caller(self, settings):
        '''
        stacklevel matters: a warning blamed on library code tells the user
        nothing about which of their lines to change.
        '''
        with pytest.warns(DeprecationWarning) as caught:
            OpenAIEmbeddingGenerator(settings)

        assert Path(caught[0].filename).name == 'test_deprecation.py'


class TestDelegates:
    def test_the_generator_holds_a_provider(self, settings):
        generator = OpenAIEmbeddingGenerator(settings)

        assert isinstance(generator.provider, EmbeddingProvider)

    def test_the_chat_class_holds_a_provider(self, settings):
        gpt = OpenAIGPT(settings)

        assert isinstance(gpt.provider, GenerationProvider)

    def test_the_client_is_the_providers(self, settings):
        generator = OpenAIEmbeddingGenerator(settings)

        assert generator.client is generator.provider.client

    def test_replacing_the_client_reaches_the_provider(
        self, settings, stub_client
    ):
        '''
        The proxy is not decoration: anything patching .client must actually
        change what the delegated call uses.
        '''
        generator = OpenAIEmbeddingGenerator(settings)
        generator.client = stub_client

        assert generator.provider.client is stub_client

    def test_embedding_work_runs_through_the_provider(
        self, settings, stub_client
    ):
        generator = OpenAIEmbeddingGenerator(settings)
        generator.client = stub_client
        generator.load_text(['a'])
        generator.calculate_embeddings()

        assert stub_client.embeddings.models == [generator.provider.model]


class TestVendorSdkIsConfined:
    def test_only_the_provider_layer_imports_a_vendor_sdk(self):
        '''
        The layer exists so nothing else has to know which vendor is behind a
        call. A stray import elsewhere is the design quietly eroding.
        '''
        vendors = {'openai', 'anthropic', 'sentence_transformers', 'google'}
        offenders = []

        for path in PACKAGE.rglob('*.py'):
            if path.parent.name == 'llm':
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name.split('.')[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or '').split('.')[0]]
                else:
                    continue
                if vendors & set(names):
                    offenders.append(
                        f'{path.relative_to(PACKAGE)}:{node.lineno}'
                    )

        assert offenders == []
