'''Offline answer-support comparison; synthetic embeddings do not measure live model quality.'''

import argparse
import json
import math
import tempfile
from pathlib import Path

from osintgpt import Project, index_project
from osintgpt.evaluation import Question, QuestionResult
from osintgpt.graph import Edge, Entity, assertion_context, graph_for, query_graph
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot
from osintgpt.search import hybrid_search
from osintgpt.vector_store import store_for

QUALIFICATIONS = ('relation_polarity', 'assertion', 'attributed_to', 'event_time', 'statement_time')


class FixtureEmbedder:
    model = 'graph-passage-fixture'
    supports_images = False
    vocabulary = ('alpha', 'bridge', 'harbor', 'delta', 'echo', 'iota', 'kappa',
                  'nube', 'sol', '北方公司', '南方公司', 'alex', 'clinic', 'firm', 'weather')

    def embed(self, texts, **kwargs):
        vectors = []
        for text in texts:
            counts = [float(text.casefold().count(word)) for word in self.vocabulary]
            norm = math.sqrt(sum(value * value for value in counts)) or 1.0
            vectors.append([value / norm for value in counts])
        return vectors


def prepare(root, fixture):
    project = Project.create('Passage evaluation', home=root).with_settings(graph_enabled=True)
    edges = {}
    for item in fixture['documents']:
        path = project.paths.root / item['ref']
        path.write_text(item['text'], encoding='utf-8')
        Corpus.load(project.paths.sources).register(item['ref'])
        if 'source' not in item:
            continue
        snapshot = load_snapshot(path, ref=item['ref'], cache_dir=project.paths.extracts)
        quote = item.get('quote', item['text'])
        provenance = snapshot.provenance(snapshot.document(item['ref']), quote)
        provenance['supporting'] = [{'quote': quote, 'matches': provenance['matches']}]
        edge = Edge(item['source'], item['target'], item['relation'], item['ref'], quote,
                    source_id=item.get('source_id', item['source'].casefold()), target_id=item['target'].casefold(),
                    evidence_quotes=(quote,), provenance=provenance,
                    **{name: item[name] for name in QUALIFICATIONS if name in item})
        edges[item['ref']] = edge
        with graph_for(project) as graph:
            graph.add([Entity(edge.source_id, edge.source), Entity(edge.target_id, edge.target)], [edge])
    report = index_project(project, FixtureEmbedder())
    if report.failed:
        raise ValueError(f'Fixture indexing failed: {report.failed}')
    return project, edges


def score(expected, passages, gold):
    retrieved = list(dict.fromkeys(row['ref'] for row in passages))
    question = Question('Fixture', expected)
    metric = QuestionResult(question, retrieved)
    supported = [ref for ref in expected if any(row['ref'] == ref and
                 gold[ref] in row['text'] for row in passages)]
    return {'recall': metric.recall if expected else None,
            'complete_answer_support': set(supported) == set(expected) if expected else None,
            'supported_expected': supported,
            'unrelated_passages': sum(row['ref'] not in expected for row in passages),
            'passages': len(passages), 'characters': sum(len(row['text']) for row in passages)}


def run(fixture_path=None):
    fixture_path = fixture_path or Path(__file__).with_suffix('.json')
    fixture = json.loads(Path(fixture_path).read_text(encoding='utf-8'))
    gold = {item['ref']: item.get('quote', item['text']) for item in fixture['documents']}
    with tempfile.TemporaryDirectory(prefix='graph-passages-') as root:
        project, edges = prepare(root, fixture)
        results = []
        with store_for(project) as store:
            for item in fixture['questions']:
                hits = hybrid_search(project, item['query'], FixtureEmbedder(), top_k=2,
                                     terms=item['terms'], store=store)
                baseline = [{'ref': hit.ref, 'text': hit.text} for hit in hits]
                graph = query_graph(project, item['entity'], item.get('target'),
                                    direction=item['direction'], depth=4 if item.get('target') else 1, limit=6)
                claims = graph.get('path', graph.get('claims', []))
                contexts = [assertion_context(project, claim['assertion_id'], context_chars=200,
                                              limit_chars=1000) for claim in claims]
                additions = [{'ref': row['ref'], 'text': row['text']} for row in contexts if row['status'] == 'ok']
                combined = list({(row['ref'], row['text']): row for row in [*baseline, *additions]}.values())
                priority = []
                for row in [*additions, *baseline]:
                    if row['ref'] not in {selected['ref'] for selected in priority}:
                        priority.append(row)
                priority = priority[:2]
                integrity = all(
                    claim['assertion_id'] in claim['citation'] and claim['ref'] in claim['citation']
                    and claim['source_id'] == edges[claim['ref']].source_id
                    and claim['target_id'] == edges[claim['ref']].target_id
                    and all(claim[name] == getattr(edges[claim['ref']], name) for name in QUALIFICATIONS)
                    and claim['verification']['snapshot'] == 'found' for claim in claims)
                results.append({'name': item['name'], 'expected': item['expected'],
                    'baseline_refs': [row['ref'] for row in baseline],
                    'graph_refs': [row['ref'] for row in additions],
                    'baseline': score(item['expected'], baseline, gold),
                    'assisted': score(item['expected'], combined, gold),
                    'graph_first_same_passage_count': score(item['expected'], priority, gold),
                    'graph_status': graph['status'], 'expected_status': item['graph_status'],
                    'claims': len(claims), 'citation_direction_qualification_checks': integrity,
                    'relevance_not_implied': [row['ref'] for row in additions if row['ref'] not in item['expected']]})
    answerable = [row for row in results if row['expected']]
    summary = {'questions': len(results), 'answerable': len(answerable),
               'status_checks_passed': sum(row['graph_status'] == row['expected_status'] for row in results),
               'evidence_checks_passed': sum(row['citation_direction_qualification_checks'] for row in results)}
    for variant in ('baseline', 'assisted', 'graph_first_same_passage_count'):
        summary[variant] = {
            'mean_recall': sum(row[variant]['recall'] for row in answerable) / len(answerable),
            'complete_answer_support': sum(row[variant]['complete_answer_support'] for row in answerable),
            'unrelated_passages': sum(row[variant]['unrelated_passages'] for row in answerable),
            'characters': sum(row[variant]['characters'] for row in results)}
    return {'scope': 'Controlled answer-support proxy with synthetic embeddings and fixed graph calls. '
                     'No extraction model, live answering model, or natural-language planning was evaluated. '
                     'Assisted context adds passages, so its budget is larger. The graph-first comparison '
                     'keeps two passages but not an equal token budget; it tests naive priority, not a trained ranker.',
            'policy': 'Keep semantic/lexical ranking unchanged. Read graph sources explicitly; '
                      'require representative equal-budget live answer evaluation before automatic ranking integration.',
            'summary': summary, 'results': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
