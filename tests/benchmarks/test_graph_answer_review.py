'''Keep human judgments bound to the model, prompt and output limit of each answer.'''

from copy import deepcopy

import pytest

from osintgpt.graph.answer_evaluation import answer_fingerprint, pack_context, review_template, score_answer_reviews


def reviewed(answers):
    review = review_template(answers)
    review.update(reviewer='Fixture reviewer', reviewed_at='2026-10-09')
    for row in review['answers']:
        row.update(supported_claims=1, total_claims=1, correct_citations=1, total_citations=1,
                   expected_facts_covered=1, expected_facts_total=1, qualifications_preserved=True)
    return review


@pytest.fixture
def answers():
    context = pack_context([{'citation': 'a.md', 'text': 'Alpha funds Beta.'}], 400)
    return [{'question_id': 'funding', 'variant': arm, 'question': 'Who funds Beta?', 'context': context,
             'answer': 'Alpha [1].', 'system_input': 'Fixture instructions and evidence.',
             'answering_model': {'provider': 'offline', 'model': 'answer-A'}, 'output_token_budget': 128}
            for arm in ('hybrid', 'graph_assisted')]


@pytest.mark.parametrize('field,value', [('answering_model', {'provider': 'offline', 'model': 'answer-B'}),
                                        ('system_input', 'Changed instructions.'), ('output_token_budget', 256)])
def test_changed_model_prompt_or_budget_invalidates_original_review(answers, field, value):
    review = reviewed(answers)
    changed = deepcopy(answers)
    changed[0][field] = value
    assert answer_fingerprint(changed[0]) != answer_fingerprint(answers[0])
    with pytest.raises(ValueError, match='unchanged'):
        score_answer_reviews(changed, review)


@pytest.mark.parametrize('field,value', [('answering_model', {'provider': 'offline', 'model': 'answer-B'}),
                                        ('output_token_budget', 256)])
def test_even_reviewed_pairs_require_same_answer_model_and_output_limit(answers, field, value):
    answers[1][field] = value
    with pytest.raises(ValueError, match='same answering model and output limit'):
        score_answer_reviews(answers, reviewed(answers))
