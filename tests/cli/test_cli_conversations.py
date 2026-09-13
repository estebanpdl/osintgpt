'''Threads through the CLI: continuing one, and managing what is stored.'''

import json

from osintgpt.projects import list_conversations

from test_cli_retrieval import (
    home,
    invoke,
    project_with_chunks,
    runner,
    stub_providers
)


def payload(result):
    assert result.exit_code == 0, result.output

    return json.loads(result.output)


class TestAskIsStillOneShotByDefault:
    '''
    A command that silently began accumulating state would change what every
    existing script does. Threads are opted into.
    '''

    def test_no_conversation_is_written(self, runner, home, monkeypatch):
        project = project_with_chunks(runner, home)
        stub_providers(monkeypatch)

        invoke(runner, home, 'ask', 'What is alpha?', '--project',
               'case-search')

        assert list_conversations(project) == []

    def test_no_conversation_key_in_the_json(self, runner, home, monkeypatch):
        project_with_chunks(runner, home)
        stub_providers(monkeypatch)

        data = payload(invoke(
            runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
            '--json'
        ))

        assert 'conversation' not in data


class TestStartingAndContinuingAThread:
    def test_new_conversation_records_the_turn(
        self, runner, home, monkeypatch
    ):
        project = project_with_chunks(runner, home)
        stub_providers(monkeypatch)

        data = payload(invoke(
            runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
            '--new-conversation', '--json'
        ))

        threads = list_conversations(project)
        assert len(threads) == 1
        assert data['conversation'] == threads[0].id

    def test_continuing_appends_to_the_same_thread(
        self, runner, home, monkeypatch
    ):
        project = project_with_chunks(runner, home)
        stub_providers(monkeypatch)

        first = payload(invoke(
            runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
            '--new-conversation', '--json'
        ))
        invoke(
            runner, home, 'ask', 'And the second?', '--project',
            'case-search', '--conversation', first['conversation'], '--json'
        )

        threads = list_conversations(project)
        assert len(threads) == 1

        shown = payload(invoke(
            runner, home, 'conversations', 'show', first['conversation'],
            '--project', 'case-search', '--json'
        ))
        assert [t['question'] for t in shown['turns']] == [
            'What is alpha?', 'And the second?'
        ]

    def test_an_unknown_thread_fails_rather_than_starting_one(
        self, runner, home, monkeypatch
    ):
        project = project_with_chunks(runner, home)
        stub_providers(monkeypatch)

        result = invoke(
            runner, home, 'ask', 'q', '--project', 'case-search',
            '--conversation', 'no-such-thread'
        )

        assert result.exit_code != 0
        assert list_conversations(project) == []


class TestManagingThreads:
    def test_listing_is_empty_before_anything_is_asked(
        self, runner, home, monkeypatch
    ):
        project_with_chunks(runner, home)

        data = payload(invoke(
            runner, home, 'conversations', 'list', '--project',
            'case-search', '--json'
        ))

        assert data['conversations'] == []

    def test_listing_names_the_thread(self, runner, home, monkeypatch):
        project_with_chunks(runner, home)
        stub_providers(monkeypatch)
        invoke(runner, home, 'ask', 'What is alpha?', '--project',
               'case-search', '--new-conversation', '--json')

        data = payload(invoke(
            runner, home, 'conversations', 'list', '--project',
            'case-search', '--json'
        ))

        assert data['conversations'][0]['title'] == 'What is alpha?'

    def test_show_carries_the_answer_and_its_sources(
        self, runner, home, monkeypatch
    ):
        project_with_chunks(runner, home)
        stub_providers(monkeypatch)
        started = payload(invoke(
            runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
            '--new-conversation', '--json'
        ))

        shown = payload(invoke(
            runner, home, 'conversations', 'show', started['conversation'],
            '--project', 'case-search', '--json'
        ))

        assert shown['turns'][0]['answer'] == started['answer']
        assert shown['turns'][0]['sources'] == started['sources']

    def test_delete_removes_it(self, runner, home, monkeypatch):
        project = project_with_chunks(runner, home)
        stub_providers(monkeypatch)
        started = payload(invoke(
            runner, home, 'ask', 'What is alpha?', '--project', 'case-search',
            '--new-conversation', '--json'
        ))

        invoke(
            runner, home, 'conversations', 'delete', started['conversation'],
            '--project', 'case-search', '--yes', '--json'
        )

        assert list_conversations(project) == []

    def test_showing_an_unknown_thread_fails(self, runner, home):
        project_with_chunks(runner, home)

        result = invoke(
            runner, home, 'conversations', 'show', 'no-such-thread',
            '--project', 'case-search'
        )

        assert result.exit_code != 0
