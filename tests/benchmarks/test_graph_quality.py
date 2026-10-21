'''Exercise scoring failures independently of annotation replay and preserve review gates.'''

import json
from copy import deepcopy
from pathlib import Path

import pytest

from osintgpt.graph.evaluation import corpus_digest, load_corpus, review_status, score_case, score_extractions
from osintgpt.graph.answer_evaluation import answer_fingerprint, citation_bounds, pack_context, review_template, score_answer_reviews
from graph_quality import run

CORPUS = Path(__file__).with_name('graph_quality.json')


@pytest.fixture(scope='module')
def corpus():
    return load_corpus(CORPUS)


@pytest.fixture(scope='module')
def replay(corpus):
    return run(corpus)


def case(corpus, name):
    return next(item for item in corpus['cases'] if item['id'] == name)


def test_real_builder_replay_preserves_annotations_without_measuring_model_quality(replay, corpus):
    assert replay['execution'] == 'annotation_replay'
    assert replay['annotation_review']['status'] == 'pending'
    assert replay['summary']['failed_cases'] == 0
    assert replay['summary']['entities']['true_positive'] == 33
    assert replay['summary']['qualified_relationships']['true_positive'] == 16
    assert replay['summary']['aliases']['true_positive'] == 1
    assert set(replay['by_language']) == {'en', 'es', 'zh', 'ar'}
    scanned = replay['predictions']['scan-cache']['edges'][0]
    assert scanned['provenance']['matches'][0]['spans'][0]['page'] == 1
    for name in ('csv-record', 'json-record'):
        assert replay['predictions'][name]['edges'][0]['ref'].endswith('#r1')


@pytest.mark.parametrize('change', ['relation_polarity', 'assertion', 'attributed_to', 'event_time', 'statement_time', 'quote', 'direction', 'ref'])
def test_scorer_detects_authentic_but_incorrect_claims(corpus, replay, change):
    gold = case(corpus, 'es-denial')
    predicted = deepcopy(replay['predictions'][gold['id']])
    edge = predicted['edges'][0]
    if change == 'quote':
        edge['evidence_quotes'] = ['Nube']
    elif change == 'direction':
        edge['source_id'], edge['target_id'] = edge['target_id'], edge['source_id']
    elif change == 'ref':
        edge['ref'] = 'other.md'
    else:
        edge[change] = 'incorrect'
    scored = score_case(gold, predicted)
    assert scored['qualified_relationships']['true_positive'] == 0
    assert scored['qualified_relationships']['false_positive'] == 1
    if change == 'quote':
        assert scored['evidence']['rate'] == 1 and not scored['claims'][0]['gold_excerpt_covered']


def test_duplicates_and_namesake_merges_cannot_inflate_scores(corpus, replay):
    gold = case(corpus, 'namesakes')
    predicted = deepcopy(replay['predictions'][gold['id']])
    predicted['edges'].append(deepcopy(predicted['edges'][0]))
    assert score_case(gold, predicted)['relationships']['false_positive'] == 1
    for entity in predicted['entities']:
        if entity['name'] == 'Alex':
            entity['provenance'] = {}
    scored = score_case(gold, predicted)
    assert scored['relationships']['true_positive'] == 0
    assert scored['entities']['false_negative'] == 2


def test_incomplete_predictions_cannot_consume_a_later_correct_match(corpus, replay):
    gold = case(corpus, 'conflict')
    predicted = deepcopy(replay['predictions'][gold['id']])
    incomplete = deepcopy(predicted['edges'][0])
    incomplete.update(relation_polarity='unknown', attributed_to='')
    correct = next(edge for edge in predicted['edges'] if edge['relation_polarity'] == 'positive')
    for edges in ([incomplete, correct], [correct, incomplete]):
        predicted['edges'] = edges
        assert score_case(gold, predicted)['qualified_relationships']['true_positive'] == 1


def test_missing_predictions_remain_failures_and_empty_controls_do_not_score_perfect(corpus, replay):
    empty = score_extractions(corpus, {})
    assert empty['summary']['failed_cases'] == len(corpus['cases'])
    assert empty['summary']['relationships']['recall'] == 0
    control = score_case(case(corpus, 'no-relationship'), replay['predictions']['no-relationship'])
    assert control['relationships']['precision'] is None and control['relationships']['recall'] is None
    with pytest.raises(ValueError, match='unknown'):
        score_extractions(corpus, {'typo': {}})


def test_review_is_bound_to_annotations_and_source(corpus):
    reviewed = {'corpus_sha256': corpus_digest(corpus), 'reviewer': 'Fixture reviewer', 'reviewed_at': '2026-10-08',
                'approved': True, 'case_ids': [item['id'] for item in corpus['cases']]}
    assert review_status(corpus, reviewed)['status'] == 'reviewed'
    changed = deepcopy(corpus)
    changed['cases'][0]['edges'][0]['relations'].append('new meaning')
    assert review_status(changed, reviewed)['status'] == 'pending'


def test_loader_rejects_invalid_annotations_and_escaping_sources(corpus, tmp_path):
    for field, value in [('ref', '../outside.md'), ('text', 'Quotation absent.')]:
        changed = deepcopy(corpus)
        changed['cases'][0][field] = value
        path = tmp_path / 'corpus.json'
        path.write_text(json.dumps(changed), encoding='utf8')
        with pytest.raises(ValueError):
            load_corpus(path)


def test_equal_input_budget_keeps_whole_multilingual_evidence():
    evidence = [{'citation': 'a.md', 'text': '语' * 3000}, {'citation': 'b.md', 'text': 'Álvaro no financió a Belén.'}]
    packed = pack_context(evidence, 400, question='¿Qué ocurrió?')
    assert packed['tokens'] <= 400 and packed['omitted'] == ['a.md']
    assert packed['passages'] == evidence[1:]
    assert citation_bounds('[1] supported; [2] absent.', packed['passages'])['out_of_range'] == [2]
    with pytest.raises(ValueError, match='shared instructions'):
        pack_context(evidence, 1)


def test_answer_review_rejects_unreviewed_modified_and_unpaired_answers():
    context = pack_context([{'citation': 'a.md', 'text': 'Alpha funds Beta.'}], 400)
    answers = [{'question_id': 'q1', 'variant': variant, 'question': 'Who funds Beta?',
                'answer': 'Alpha [1].', 'context': context} for variant in ('hybrid', 'graph_assisted')]
    review = review_template(answers)
    with pytest.raises(ValueError, match='reviewer'):
        score_answer_reviews(answers, review)
    review.update(reviewer='Fixture reviewer', reviewed_at='2026-10-08')
    with pytest.raises(ValueError, match='nonnegative'):
        score_answer_reviews(answers, review)
    for row in review['answers']:
        row.update(supported_claims=1, total_claims=1, correct_citations=1, total_citations=1,
                   expected_facts_covered=1, expected_facts_total=1, qualifications_preserved=True)
    assert score_answer_reviews(answers, review)['variants']['hybrid']['citation_accuracy'] == 1
    changed = deepcopy(answers)
    changed[0]['answer'] = 'Invented [1].'
    assert answer_fingerprint(changed[0]) != answer_fingerprint(answers[0])
    with pytest.raises(ValueError, match='unchanged'):
        score_answer_reviews(changed, review)
    review['answers'] = review['answers'][:1]
    with pytest.raises(ValueError, match='both comparison'):
        score_answer_reviews(answers[:1], review)


def test_paired_runner_uses_shared_services_and_saves_exact_inputs(tmp_path):
    from graph_answers import run as run_answers
    from graph_passages import FixtureEmbedder
    from osintgpt.llm.usage import Usage, UsageRecorder

    recorder = UsageRecorder()
    class Generator:
        provider, model = 'offline', 'scripted'

        def __init__(self):
            self.recorder, self.inputs = recorder, []

        def generate(self, system, user, *, max_output_tokens=None):
            self.inputs.append((system, user, max_output_tokens))
            recorder.record(Usage(self.provider, self.model, 10, 5, billable=False))
            if user.startswith('{'):
                return json.dumps({'entities': [{'id': 'a', 'name': 'Alpha', 'evidence': ['Alpha']},
                                                {'id': 'b', 'name': 'Beta', 'evidence': ['Beta']}],
                                   'edges': [{'source': 'a', 'target': 'b', 'relation': 'funds',
                                              'evidence': ['Alpha funds Beta.']}]})
            return 'Alpha [1].'

    fixture = {'documents': [{'ref': 'a.md', 'text': 'Alpha funds Beta.'}],
               'questions': [{'name': 'funding', 'query': 'Who funds Beta?', 'terms': ['Alpha'],
                              'entity': 'Alpha', 'target': 'Beta', 'direction': 'outgoing', 'expected': ['a.md']}]}
    generator, output = Generator(), tmp_path / 'answers.json'
    report = run_answers(fixture, generator, FixtureEmbedder(), recorder, budget=1800, output_tokens=128, output=output)
    assert report['status'] == 'complete_pending_review' and len(report['answers']) == 2, report.get('problem')
    assert report['review_template']['reviewer'] == ''
    assert report['graph_build']['status'] == 'complete'
    assert {row['variant'] for row in report['answers']} == {'hybrid', 'graph_assisted'}
    assert all(row['context']['tokens'] <= 1800 for row in report['answers'])
    assert all(limit == 128 for _, user, limit in generator.inputs if not user.startswith('{'))
    assisted = next(row for row in report['answers'] if row['variant'] == 'graph_assisted')
    assert assisted['graph_status']['path_complete']
    assert json.loads(output.read_text(encoding='utf8'))['answers'] == report['answers']


@pytest.mark.parametrize('budget', ['nan', 'inf', '0', '-1'])
@pytest.mark.parametrize('module', ['graph_quality', 'graph_answers'])
def test_live_entrypoints_require_finite_positive_ceiling_before_clients(monkeypatch, tmp_path, module, budget):
    import importlib
    runner = importlib.import_module(module)
    def forbidden(*args, **kwargs):
        raise AssertionError('No provider may be constructed for an invalid budget')
    monkeypatch.setattr(runner, 'build_generation_provider', forbidden)
    argv = [module, '--output', str(tmp_path / 'result.json'), '--provider', 'openai',
            '--model', 'unused', '--max-cost-usd', budget]
    argv += ['--live']
    if module == 'graph_answers':
        argv += ['--embedding-provider', 'openai', '--embedding-model', 'unused']
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
