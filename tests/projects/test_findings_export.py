'''Selected findings round-trip citations, original content, and answer-time evidence.'''

import csv
import io
import json

import pytest

from osintgpt import Project
from osintgpt.findings import canon_pages, findings_package, findings_markdown, findings_csv
from osintgpt.projects import start_conversation, append_turn


@pytest.fixture
def saved(tmp_path):
    project = Project.create('Findings', home=tmp_path)
    conversation = start_conversation(project, '=Résumé, ¿Quién?')
    claim = {'assertion_id': 'assert_example', 'source': 'Álex', 'target': 'Бета', 'relation': '资助',
             'citation': 'a.md [assert_example]', 'ref': 'a.md', 'source_version': 'version-1',
             'relation_polarity': 'negative', 'assertion': 'alleged', 'event_time': '2024', 'attributed_to': 'X',
             'evidence_quotes': ['Álex no financió Бета.', 'X lo afirma.'],
             'verification': {'snapshot': 'found', 'current': 'not_found'},
             'provenance': {'source_version': 'version-1', 'matches': [{'start': 1, 'end': 22, 'spans': [{'page': 9}]}]}}
    answer = {'answer': '+Según X, Álex no financió Бета [a.md; assert_example].\nSecond line.', 'sources': ['a.md'],
              'trace': {'calls': [{'tool': 'graph_query', 'arguments': {'entity': 'alex'}, 'payload': {
                  'claims': [claim], 'graph': {'stale': True, 'incomplete': True, 'build_finished_at': '2026-10-07'},
                  'truncated': True, 'truncation_reasons': ['result_limit']}},
                  {'tool': 'exact_search', 'payload': {'passages': [{'citation': 'b.md#section', 'ref': 'b.md',
                    'text': '語' * 1000, 'provenance': {'source_version': 'version-2'}}]}}]}}
    append_turn(conversation, '=Résumé, ¿Quién?', answer)
    append_turn(conversation, 'UNSELECTED QUESTION', {'answer': 'UNSELECTED ANSWER', 'sources': ['private.md']})
    (project.paths.canon / 'finding.md').write_text('# Finding\n\nOriginal [source](../a.md) and [[a]].\n', encoding='utf8')
    return project, conversation, answer


def test_selected_content_and_evidence_survive_markdown_and_csv(saved):
    project, conversation, answer = saved
    package = findings_package(project, conversation=conversation, turns=[0], pages=['finding.md'])
    markdown = findings_markdown(package)
    csv_text = findings_csv(package)
    assert 'UNSELECTED' not in markdown + csv_text and 'private.md' not in markdown + csv_text
    assert answer['answer'] in markdown and '[[a]]' in markdown and '../a.md' in markdown
    assert 'Álex no financió Бета.' in markdown and '語' * 1000 in markdown
    rows = list(csv.DictReader(io.StringIO(csv_text.lstrip('\ufeff'))))
    assert len(rows) == 2
    escaped = json.loads(rows[0]['spreadsheet_escaped_fields'])
    assert set(escaped) == {'title', 'text'}
    assert rows[0]['text'][1:] == answer['answer']
    evidence = json.loads(rows[0]['evidence'])
    claim = evidence[0]['result']['claims'][0]
    assert claim['relation_polarity'] == 'negative' and claim['verification']['current'] == 'not_found'
    assert claim['provenance']['matches'][0]['spans'][0]['page'] == 9
    assert json.loads(rows[0]['graph_states'])[0]['incomplete']
    assert json.loads(rows[0]['citations']) == ['a.md']
    assert rows[1]['source_version'].startswith('sha256:')
    assert findings_csv(package) == findings_csv(package)
    assert findings_markdown(package) == findings_markdown(findings_package(project, conversation=conversation, turns=[0], pages=['finding.md']))


def test_older_answers_remain_exportable_with_explicit_missing_evidence(saved):
    project, conversation, _ = saved
    package = findings_package(project, conversation=conversation, turns=[1])
    assert package['items'][0]['evidence_status'] == 'not_recorded'
    assert 'No tool evidence was recorded' in findings_markdown(package)
    assert json.loads(list(csv.DictReader(io.StringIO(findings_csv(package).lstrip('\ufeff'))))[0]['citations']) == ['private.md']


@pytest.mark.parametrize('page', ['../project.toml', '../outside.md', 'missing.md'])
def test_finding_paths_cannot_escape_canon(saved, page):
    project, _, _ = saved
    with pytest.raises(ValueError):
        findings_package(project, pages=[page])
    assert canon_pages(project) == ['finding.md']


def test_empty_and_cross_project_selections_are_refused(saved, tmp_path):
    project, conversation, _ = saved
    with pytest.raises(ValueError, match='at least one'):
        findings_package(project)
    with pytest.raises(ValueError, match='existing answer'):
        findings_package(project, conversation=conversation, turns=[99])
    other = Project.create('Other', home=tmp_path)
    with pytest.raises(ValueError, match='different project'):
        findings_package(other, conversation=conversation, turns=[0])
