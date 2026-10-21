'''Annotation-based graph scoring with conservative identity alignment and explicit unreviewed results.'''

import hashlib
import json
from pathlib import Path

from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import quote_locations, text_spans
from .records import merge_key
from .contract import ASSERTIONS, POLARITIES

QUALIFIERS = ('relation_polarity', 'assertion', 'attributed_to', 'event_time', 'statement_time')


def _document(case):
    return Document(case['ref'], case['text'], spans=tuple(text_spans(case['text'])))


def corpus_digest(corpus):
    return hashlib.sha256(json.dumps(corpus, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()


def load_corpus(path):
    corpus = json.loads(Path(path).read_text(encoding='utf8'))
    if corpus.get('format_version') != 1 or not corpus.get('cases'):
        raise ValueError('Expected graph benchmark format 1 with nonempty cases.')
    seen = set()
    for case in corpus['cases']:
        if case['id'] in seen:
            raise ValueError('Benchmark case IDs must be unique.')
        seen.add(case['id'])
        source = Path(case['ref'].split('#', 1)[0])
        if source.is_absolute() or '..' in source.parts or len(source.parts) != 1 or '\\' in str(source) or ':' in str(source):
            raise ValueError('Benchmark sources must be filenames inside the isolated project.')
        document = _document(case)
        identities = {entity['id'] for entity in case['entities']}
        if len(identities) != len(case['entities']):
            raise ValueError('Gold entity IDs must be unique within each case.')
        for entity in case['entities']:
            if not entity['names'] or not entity['anchors']:
                raise ValueError('Gold identities require names and source anchors.')
            for anchor in entity['anchors']:
                if not quote_locations(document, anchor):
                    raise ValueError(f"{case['id']}: identity anchor absent from source.")
        for edge in case['edges']:
            if edge['source'] not in identities or edge['target'] not in identities or not edge['relations']:
                raise ValueError('Gold relationship has an invalid endpoint or relation set.')
            if not edge['evidence'] or any(not quote_locations(document, q) for q in edge['evidence']):
                raise ValueError(f"{case['id']}: gold quotation absent from source.")
            if set(edge['qualifiers']) != set(QUALIFIERS):
                raise ValueError('Gold relationships must specify every qualification, including unknown/empty values.')
            if edge['qualifiers']['relation_polarity'] not in POLARITIES or edge['qualifiers']['assertion'] not in ASSERTIONS:
                raise ValueError('Gold relation_polarity or assertion type is unsupported.')
    return corpus


def review_status(corpus, review=None):
    review = review or {}
    approved = (review.get('corpus_sha256') == corpus_digest(corpus) and bool(review.get('reviewer'))
                and bool(review.get('reviewed_at')) and review.get('approved') is True
                and set(review.get('case_ids', [])) == {case['id'] for case in corpus['cases']})
    return {'status': 'reviewed' if approved else 'pending', 'record': review}


def _metric(tp, predicted, expected):
    precision = tp / predicted if predicted else None
    recall = tp / expected if expected else None
    return {'true_positive': tp, 'false_positive': predicted - tp, 'false_negative': expected - tp,
            'predicted': predicted, 'expected': expected, 'precision': precision, 'recall': recall,
            'f1': 2 * tp / (predicted + expected) if predicted + expected else None}


def _align(case, entities):
    aligned, details, used = {}, [], set()
    for entity in entities:
        names = {merge_key(name) for name in [entity['name'], *entity.get('aliases', [])]}
        candidates = [gold for gold in case['entities'] if names & {merge_key(n) for n in gold['names']}]
        if len(candidates) > 1:
            provenance = entity.get('provenance', {})
            origins = [provenance, *provenance.get('identity_sources', [])]
            ranges = [m for origin in origins if origin.get('document_ref') == case['ref']
                      for m in origin.get('matches', [])]
            supported = ' '.join(case['text'][m['start']:m['end']] for m in ranges)
            candidates = [gold for gold in candidates if any(anchor in supported for anchor in gold['anchors'])]
        target = candidates[0]['id'] if len(candidates) == 1 else None
        key = entity['key']
        if target in used or key in aligned:
            target = None
        if target is not None:
            aligned[key] = target
            used.add(target)
        details.append({'predicted_id': key, 'gold_id': target,
                        'candidates': [gold['id'] for gold in candidates]})
    return aligned, details


def _matching(candidates):
    owners = {}
    def place(predicted, seen):
        for gold in candidates[predicted]:
            if gold in seen:
                continue
            seen.add(gold)
            if gold not in owners or place(owners[gold], seen):
                owners[gold] = predicted
                return True
        return False
    for index in range(len(candidates)):
        place(index, set())
    return {predicted: gold for gold, predicted in owners.items()}


def score_case(case, prediction):
    entities, edges = prediction.get('entities', []), prediction.get('edges', [])
    aligned, details = _align(case, entities)
    candidates, strict, checks, evidence_valid = [], [], [], 0
    field_candidates = {name: [] for name in QUALIFIERS}
    document = _document(case)
    for edge in edges:
        quotes = edge.get('evidence_quotes') or ([edge['evidence']] if edge.get('evidence') else [])
        valid = bool(quotes) and all(quote_locations(document, quote) for quote in quotes)
        valid = valid and edge.get('ref') == case['ref']
        evidence_valid += int(valid)
        possible = [(i, gold) for i, gold in enumerate(case['edges'])
                      if aligned.get(edge.get('source_id')) == gold['source']
                      and aligned.get(edge.get('target_id')) == gold['target']
                      and merge_key(edge['relation']) in {merge_key(r) for r in gold['relations']}
                      and edge.get('ref') == case['ref']]
        candidates.append([i for i, gold in possible])
        strict.append([])
        evidence = {'assertion_id': edge.get('assertion_id', ''), 'evidence_valid': valid, 'candidates': {}}
        locations = [match for quote in quotes for match in quote_locations(document, quote)]
        for name in QUALIFIERS:
            field_candidates[name].append([i for i, gold in possible if edge.get(name, '') == gold['qualifiers'][name]])
        for index, gold in possible:
            correct = [name for name in QUALIFIERS if edge.get(name, '') == gold['qualifiers'][name]]
            support = all(any(actual['start'] <= expected['start'] and actual['end'] >= expected['end']
                              for actual in locations for expected in quote_locations(document, quote))
                          for quote in gold['evidence'])
            if valid and support and len(correct) == len(QUALIFIERS):
                strict[-1].append(index)
            evidence['candidates'][index] = {'gold_excerpt_covered': support,
                                            'incorrect_qualifiers': [k for k in QUALIFIERS if k not in correct]}
        checks.append(evidence)
    structural, qualified = _matching(candidates), _matching(strict)
    qualification = {name: len(_matching(rows)) for name, rows in field_candidates.items()}
    matched = []
    for index, evidence in enumerate(checks):
        gold = qualified.get(index, structural.get(index))
        matched.append({'assertion_id': evidence['assertion_id'], 'comparison_gold_index': gold,
                        'structural_gold_index': structural.get(index), 'qualified_gold_index': qualified.get(index),
                        'evidence_valid': evidence['evidence_valid'],
                        **evidence['candidates'].get(gold, {})})
    expected_aliases = {entity['id']: {merge_key(name) for name in entity['names'][1:]} for entity in case['entities']}
    predicted_aliases = [(aligned.get(entity['key']), merge_key(name)) for entity in entities for name in entity.get('aliases', [])]
    correct_aliases = {(key, name) for key, name in predicted_aliases if name in expected_aliases.get(key, set())}
    return {'id': case['id'], 'language': case['language'], 'tags': case['tags'],
            'entities': _metric(len(aligned), len(entities), len(case['entities'])),
            'aliases': _metric(len(correct_aliases), len(predicted_aliases), sum(len(v) for v in expected_aliases.values())),
            'relationships': _metric(len(structural), len(edges), len(case['edges'])),
            'qualified_relationships': _metric(len(qualified), len(edges), len(case['edges'])),
            'evidence': {'valid': evidence_valid, 'total': len(edges),
                         'rate': evidence_valid / len(edges) if edges else None},
            'qualifiers': {k: {'correct': qualification[k], 'matched': len(structural),
                               'accuracy': qualification[k] / len(structural) if structural else None} for k in QUALIFIERS},
            'alignment': details, 'claims': matched, 'failure': prediction.get('failure', '')}


def _aggregate(rows):
    result = {'cases': len(rows), 'failed_cases': sum(bool(row['failure']) for row in rows)}
    for key in ('entities', 'aliases', 'relationships', 'qualified_relationships'):
        result[key] = _metric(*(sum(row[key][field] for row in rows)
                                for field in ('true_positive', 'predicted', 'expected')))
    valid, total = (sum(row['evidence'][key] for row in rows) for key in ('valid', 'total'))
    result['evidence'] = {'valid': valid, 'total': total, 'rate': valid / total if total else None}
    result['qualifiers'] = {}
    for key in QUALIFIERS:
        correct, matched = (sum(row['qualifiers'][key][field] for row in rows) for field in ('correct', 'matched'))
        result['qualifiers'][key] = {'correct': correct, 'matched': matched, 'accuracy': correct / matched if matched else None}
    return result


def score_extractions(corpus, predictions, *, review=None, execution='external'):
    ids = {case['id'] for case in corpus['cases']}
    if set(predictions) - ids:
        raise ValueError('Predictions contain unknown benchmark case IDs.')
    rows = [score_case(case, predictions.get(case['id'], {'failure': 'No prediction recorded.'})) for case in corpus['cases']]
    return {'format_version': 1, 'corpus_sha256': corpus_digest(corpus), 'execution': execution,
            'annotation_review': review_status(corpus, review), 'summary': _aggregate(rows),
            'by_language': {language: _aggregate([row for row in rows if row['language'] == language])
                            for language in sorted({row['language'] for row in rows})},
            'by_tag': {tag: _aggregate([row for row in rows if tag in row['tags']])
                       for tag in sorted({tag for row in rows for tag in row['tags']})}, 'results': rows}
