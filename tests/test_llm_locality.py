# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_llm_locality.py
# Description: Whether a configuration keeps content on the machine, including
#   the case where a local-looking backend points somewhere else.
# =================================================================================

# import modules
import os
import subprocess
import sys
import pytest

# import submodules
from pathlib import Path

# import osintgpt config
from osintgpt.config import Settings

# import osintgpt llm
from osintgpt.llm import audit_locality
from osintgpt.llm.locality import is_loopback

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_EMBEDDING = 'sentence-transformers'


class TestIsLoopback:
    @pytest.mark.parametrize('url', [
        'http://localhost:11434',
        'http://127.0.0.1:11434',
        'http://0.0.0.0:11434',
        'localhost:11434',
        'http://[::1]:11434'
    ])
    def test_addresses_that_cannot_leave(self, url):
        assert is_loopback(url) is True

    @pytest.mark.parametrize('url', [
        'http://gpu-box.lan:11434',
        'http://192.168.1.50:11434',
        'http://ollama:11434',
        'https://ollama.example.com'
    ])
    def test_addresses_that_reach_another_host(self, url):
        assert is_loopback(url) is False


class TestVerdict:
    def test_the_local_pair_is_local(self):
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'ollama')

        assert report.is_local is True
        assert report.remote == []
        assert 'nothing leaves this machine' in report.summary

    def test_a_hosted_pair_is_not(self):
        report = audit_locality(Settings(), 'openai', 'anthropic')

        assert report.is_local is False
        assert len(report.remote) == 2

    def test_one_hosted_role_is_enough_to_break_it(self):
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'openai')

        assert report.is_local is False
        assert [p.role for p in report.remote] == ['generation']

    def test_the_summary_names_what_broke_it(self):
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'openai')

        assert 'generation' in report.summary
        assert 'openai' in report.summary

    def test_an_unknown_provider_is_caught(self):
        with pytest.raises(ValueError, match='unknown embedding provider'):
            audit_locality(Settings(), 'not-a-provider', 'ollama')


class TestOllamaHost:
    def test_the_default_host_is_local(self):
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'ollama')

        assert report.is_local is True

    def test_a_remote_ollama_is_not_local(self):
        '''
        The failure this exists to catch: the backend kind says local, but the
        base URL points at somebody else's machine.
        '''
        settings = Settings(ollama_base_url='http://gpu-box.lan:11434')
        report = audit_locality(settings, LOCAL_EMBEDDING, 'ollama')

        assert report.is_local is False
        assert 'gpu-box.lan' in report.summary

    def test_a_container_hostname_is_not_treated_as_local(self):
        settings = Settings(ollama_base_url='http://ollama:11434')
        report = audit_locality(settings, LOCAL_EMBEDDING, 'ollama')

        assert report.is_local is False


class TestSetupRequirements:
    def test_a_bare_model_name_needs_a_download(self):
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'ollama')
        joined = ' '.join(report.setup)

        assert 'download' in joined

    def test_a_path_needs_no_download(self):
        report = audit_locality(
            Settings(), LOCAL_EMBEDDING, 'ollama',
            embedding_model='/models/bge-small'
        )

        assert not any('download' in item for item in report.setup)

    def test_ollama_needs_its_model_pulled(self):
        report = audit_locality(
            Settings(), LOCAL_EMBEDDING, 'ollama', generation_model='qwen3:8b'
        )

        assert any('pull qwen3:8b' in item for item in report.setup)

    def test_hosted_providers_need_no_local_setup(self):
        report = audit_locality(Settings(), 'openai', 'anthropic')

        assert report.setup == []

    def test_being_local_is_not_the_same_as_needing_no_network(self):
        '''
        A local configuration still fetches a model once. Saying so is the
        difference between "offline after setup" and "works air-gapped".
        '''
        report = audit_locality(Settings(), LOCAL_EMBEDDING, 'ollama')

        assert report.is_local is True
        assert report.setup


class TestNetworkGuard:
    '''
    The CI offline job is only meaningful if its guard actually blocks. This
    proves it does, rather than trusting a green run.
    '''

    def run_guarded(self, snippet):
        # The environment is inherited rather than cleared: on Windows,
        # creating a socket at all needs SYSTEMROOT to initialise Winsock.
        environment = dict(os.environ, PYTHONPATH='ci')

        return subprocess.run(
            [sys.executable, '-c', snippet],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True
        )

    def test_the_guard_blocks_an_outbound_connection(self):
        result = self.run_guarded(
            'import socket\n'
            'socket.create_connection(("api.openai.com", 443), timeout=5)\n'
        )

        assert result.returncode != 0
        assert 'NetworkBlocked' in result.stderr

    def test_the_guard_leaves_loopback_alone(self):
        result = self.run_guarded(
            'import socket, sitecustomize\n'
            's = socket.socket()\n'
            'try:\n'
            '    s.connect(("127.0.0.1", 1))\n'
            'except sitecustomize.NetworkBlocked:\n'
            '    raise SystemExit("loopback was blocked")\n'
            'except OSError:\n'
            '    pass\n'
        )

        assert result.returncode == 0, result.stderr
