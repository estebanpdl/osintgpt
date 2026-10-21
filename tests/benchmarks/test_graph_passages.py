import pytest

from graph_passages import run


def test_controlled_comparison_exposes_recovered_support_and_irrelevant_additions():
    report = run()
    summary = report['summary']
    assert summary['status_checks_passed'] == summary['evidence_checks_passed'] == summary['questions']
    assert summary['assisted']['complete_answer_support'] > summary['baseline']['complete_answer_support']
    assert summary['assisted']['unrelated_passages'] > summary['baseline']['unrelated_passages']
    assert summary['assisted']['mean_recall'] == pytest.approx(1.0)
    weather = next(row for row in report['results'] if row['name'] == 'nonrelational weather')
    assert 'ownership.md' in weather['relevance_not_implied']
    assert weather['baseline']['complete_answer_support']
    assert not weather['graph_first_same_passage_count']['complete_answer_support']
    ambiguous = next(row for row in report['results'] if row['name'] == 'namesake ambiguity')
    assert ambiguous['claims'] == 0
