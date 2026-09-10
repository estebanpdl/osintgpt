'''Graph evidence verification checks source text without changing claims.'''

import unicodedata

import pytest

from osintgpt.graph import (
    Edge,
    EvidenceReport,
    graph_for,
    verify_evidence
)
from osintgpt.projects import Project


@pytest.fixture
def project(tmp_path):
    return Project.create('Evidence', path=tmp_path / 'project')


def add_edge(project, ref, evidence):
    with graph_for(project) as graph:
        graph.add([], [Edge(
            source='Alpha', target='Beta', relation='supports',
            ref=ref, evidence=evidence
        )])


def test_verbatim_evidence_passes(project):
    (project.paths.root / 'claim.md').write_text(
        'Context. Alpha supports Beta. More context.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', 'Alpha supports Beta.')

    report = verify_evidence(project)

    assert report.found == 1
    assert report.not_found == 0
    assert report.failures == []
    assert report.results[0].status == 'found'


def test_paraphrased_evidence_fails(project):
    (project.paths.root / 'claim.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', 'Beta receives support from Alpha.')

    report = verify_evidence(project)

    assert report.not_found == 1
    assert report.failures == report.results
    assert report.results[0].status == 'not_found'


def test_collapsed_whitespace_and_nfc_pass(project):
    decomposed = unicodedata.normalize('NFD', 'Café')
    (project.paths.root / 'claim.md').write_text(
        f'{decomposed}\n\t supports   Beta.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', 'Café supports Beta.')

    assert verify_evidence(project).found == 1


def test_case_changes_do_not_pass(project):
    (project.paths.root / 'claim.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', 'alpha supports Beta.')

    assert verify_evidence(project).not_found == 1


def test_a_missing_document_is_reported(project):
    add_edge(project, 'deleted.md', 'Alpha supports Beta.')

    report = verify_evidence(project)

    assert report.unreadable == 1
    assert report.failures[0].problem == 'document does not exist'


@pytest.mark.parametrize('evidence', [
    'Альфа поддерживает Бету.',
    'ألفا تدعم بيتا.',
    '阿尔法支持贝塔。'
])
def test_non_latin_evidence_verifies_like_latin(project, evidence):
    (project.paths.root / 'claim.md').write_text(evidence, encoding='utf-8')
    add_edge(project, 'claim.md', evidence)

    assert verify_evidence(project).found == 1


def test_a_graph_that_was_never_built_is_empty_and_not_created(project):
    graph_path = project.paths.root / 'graph.sqlite'

    report = verify_evidence(project)

    assert report == EvidenceReport()
    assert report.summary == 'nothing to verify'
    assert graph_path.exists() is False


def test_verification_does_not_modify_the_graph(project):
    (project.paths.root / 'claim.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', 'A paraphrase.')
    with graph_for(project) as graph:
        before = graph.edge_count

    verify_evidence(project)

    with graph_for(project) as graph:
        assert graph.edge_count == before


def test_refs_restrict_the_checked_edges(project):
    for ref in ('first.md', 'second.md'):
        (project.paths.root / ref).write_text(
            'Alpha supports Beta.', encoding='utf-8'
        )
        add_edge(project, ref, 'Alpha supports Beta.')

    report = verify_evidence(project, refs=['second.md'])

    assert report.total == 1
    assert report.results[0].edge.ref == 'second.md'


def test_summary_carries_every_count(project):
    (project.paths.root / 'found.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    (project.paths.root / 'wrong.md').write_text(
        'Something else.', encoding='utf-8'
    )
    add_edge(project, 'found.md', 'Alpha supports Beta.')
    add_edge(project, 'wrong.md', 'Alpha supports Beta.')
    add_edge(project, 'missing.md', 'Alpha supports Beta.')

    report = verify_evidence(project)

    assert report.summary == '1 verified, 1 not found, 1 unreadable'


def test_an_empty_evidence_value_cannot_verify(project):
    (project.paths.root / 'claim.md').write_text(
        'Alpha supports Beta.', encoding='utf-8'
    )
    add_edge(project, 'claim.md', '')

    assert verify_evidence(project).not_found == 1
