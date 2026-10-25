'''SDK transport compatibility without installing a different provider version.'''

import builtins
import importlib.util
import sys

import httpx
from openai import APIConnectionError
import pytest

from osintgpt.llm import embedding_errors


def load_classifier(monkeypatch, missing=None):
    original = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == 'httpx2' and missing:
            raise ModuleNotFoundError('simulated missing transport dependency', name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', refuse)
    name = 'osintgpt.llm._transport_import_probe'
    spec = importlib.util.spec_from_file_location(name, embedding_errors.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_classifier_accepts_sdk_environments_without_httpx2(monkeypatch):
    # The installed SDKs are loaded first. Only the classifier's optional
    # import is simulated here, not an installation of older SDK versions.
    module = load_classifier(monkeypatch, 'httpx2')
    wrapped = APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
    wrapped.__cause__ = httpx.UnsupportedProtocol('bad URL')
    assert not module.classify(wrapped, 'openai', 0).retryable
    assert module.classify(httpx.ReadTimeout('temporary'), 'gemini', 0).retryable


def test_broken_installed_transport_is_not_silently_ignored(monkeypatch):
    with pytest.raises(ModuleNotFoundError) as error:
        load_classifier(monkeypatch, 'httpcore2')
    assert error.value.name == 'httpcore2'


def test_installed_httpx2_configuration_errors_remain_terminal():
    transport = pytest.importorskip('httpx2')
    for cause in [transport.UnsupportedProtocol('bad URL'), transport.LocalProtocolError('bad header')]:
        wrapped = APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
        wrapped.__cause__ = cause
        assert not embedding_errors.classify(wrapped, 'openai', 0).retryable
