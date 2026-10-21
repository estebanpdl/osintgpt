'''Budget-matched answer inputs and hash-bound human review for controlled graph comparisons.'''

import hashlib
import json
import re

import tiktoken

from osintgpt.prompts import prompt

REVIEW_FIELDS = ('supported_claims', 'total_claims', 'correct_citations', 'total_citations',
                 'expected_facts_covered', 'expected_facts_total')


def pack_context(passages, token_budget, *, question='', encoding='cl100k_base'):
    if type(token_budget) is not int or token_budget <= 0:
        raise ValueError('Context token budget must be a positive integer.')
    encoder = tiktoken.get_encoding(encoding)
    def count(rows):
        return len(encoder.encode(prompt('answer', question=question, passages=rows), disallowed_special=())) + len(encoder.encode(question, disallowed_special=()))
    if count([]) > token_budget:
        raise ValueError('Token budget does not fit the shared instructions and question.')
    selected, omitted, seen = [], [], set()
    for passage in passages:
        identity = json.dumps(passage, sort_keys=True, ensure_ascii=False)
        if identity in seen:
            continue
        seen.add(identity)
        candidate = [*selected, passage]
        if count(candidate) <= token_budget:
            selected = candidate
        else:
            omitted.append(passage['citation'])
    return {'passages': selected, 'tokens': count(selected),
            'token_budget': token_budget, 'encoding': encoding, 'omitted': omitted}


def answer_input(question, packed):
    return prompt('answer', question=question, passages=packed['passages'])


def answer_fingerprint(row):
    value = {key: row[key] for key in ('question_id', 'variant', 'question', 'answer', 'context')}
    value['expected_refs'] = row.get('expected_refs', [])
    for key in ('answering_model', 'system_input', 'output_token_budget'):
        if key in row:
            value[key] = row[key]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()


def citation_bounds(answer, passages):
    cited = [int(number) for number in re.findall(r'\[(\d+)\]', answer)]
    return {'citation_mentions': len(cited), 'out_of_range': [number for number in cited if not 1 <= number <= len(passages)],
            'semantic_correctness': 'requires_review'}


def review_template(answers):
    return {'reviewer': '', 'reviewed_at': '', 'answers': [
        {'sha256': answer_fingerprint(row), 'question_id': row['question_id'], 'variant': row['variant'],
         **dict.fromkeys(REVIEW_FIELDS), 'qualifications_preserved': None, 'notes': ''} for row in answers]}


def score_answer_reviews(answers, review):
    if not review.get('reviewer') or not review.get('reviewed_at'):
        raise ValueError('Answer scoring requires a named reviewer and review date.')
    indexed = {entry['sha256']: entry for entry in review['answers']}
    if len(indexed) != len(review['answers']) or set(indexed) != {answer_fingerprint(row) for row in answers}:
        raise ValueError('Review must cover each unchanged answer and context exactly once.')
    pairs = {}
    for row in answers:
        pair = pairs.setdefault(row['question_id'], {})
        if row['variant'] in pair:
            raise ValueError('Duplicate question/variant in answer run.')
        pair[row['variant']] = row
    if not pairs or any(set(pair) != {'hybrid', 'graph_assisted'} for pair in pairs.values()):
        raise ValueError('Each question requires both comparison variants.')
    for pair in pairs.values():
        left, right = pair['hybrid'], pair['graph_assisted']
        if left['question'] != right['question'] or left['context']['token_budget'] != right['context']['token_budget']:
            raise ValueError('Paired answers require identical questions and input limits.')
        if (left.get('answering_model') != right.get('answering_model')
                or left.get('output_token_budget') != right.get('output_token_budget')):
            raise ValueError('Paired answers require the same answering model and output limit.')
        if indexed[answer_fingerprint(left)]['expected_facts_total'] != indexed[answer_fingerprint(right)]['expected_facts_total']:
            raise ValueError('Paired answers require the same expected fact count.')
    sums = {}
    for row in answers:
        entry = indexed[answer_fingerprint(row)]
        for key in REVIEW_FIELDS:
            if type(entry.get(key)) is not int or entry[key] < 0:
                raise ValueError('Review counts must be nonnegative integers; unreviewed answers cannot be scored.')
        for part, whole in (('supported_claims', 'total_claims'), ('correct_citations', 'total_citations'),
                            ('expected_facts_covered', 'expected_facts_total')):
            if entry[part] > entry[whole]:
                raise ValueError('Reviewed numerator exceeds denominator.')
        if type(entry.get('qualifications_preserved')) is not bool:
            raise ValueError('Qualification preservation requires an explicit reviewed judgment.')
        tally = sums.setdefault(row['variant'], dict.fromkeys((*REVIEW_FIELDS, 'answers', 'qualified_answers'), 0))
        for key in REVIEW_FIELDS:
            tally[key] += entry[key]
        tally['answers'] += 1
        tally['qualified_answers'] += entry['qualifications_preserved']
    for tally in sums.values():
        tally['unsupported_claims'] = tally['total_claims'] - tally['supported_claims']
        tally['citation_accuracy'] = tally['correct_citations'] / tally['total_citations'] if tally['total_citations'] else None
        tally['fact_recall'] = tally['expected_facts_covered'] / tally['expected_facts_total'] if tally['expected_facts_total'] else None
    return {'reviewer': review['reviewer'], 'reviewed_at': review['reviewed_at'], 'variants': sums}
