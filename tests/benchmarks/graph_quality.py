'''Run the real graph builder against annotations; replay validates plumbing, live runs measure extraction.'''

import argparse
import json
import math
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from osintgpt.config import Settings
from osintgpt.graph import ExtractionOptions, build_graph, graph_for
from osintgpt.graph.evaluation import corpus_digest, load_corpus, score_extractions
from osintgpt.llm import build_generation_provider
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from graph_fixtures import AnnotationReplay, prepare_case, write_report
from graph_benchmark import extraction_metadata, usage_summary

DEFAULT_CORPUS = Path(__file__).with_suffix('.json')


def run(corpus, *, generator=None, output=None, review=None, execution=None):
    predictions, runs = {}, []
    execution = execution or ('injected_provider' if generator is not None else 'annotation_replay')
    records = []
    with tempfile.TemporaryDirectory(prefix='osintgpt-quality-') as root:
        for case in corpus['cases']:
            project = prepare_case(root, case)
            started = time.perf_counter()
            stopped = False
            provider = generator if generator is not None else AnnotationReplay(case)
            recorder = getattr(provider, 'recorder', None)
            offset = len(recorder) if recorder is not None else 0
            try:
                report = build_graph(project, provider, options=ExtractionOptions(repair_quotes=False))
                failure = '; '.join(item.problem for item in report.failed)
                counts, usage, status = asdict(report.accounting), report.usage, report.status
            except CostLimitReached as error:
                from osintgpt.graph import saved_graph_status
                state = saved_graph_status(project)
                counts, usage, status = state['counts'], state['usage'], state['status']
                failure, stopped = str(error), True
            except Exception as error:
                from osintgpt.graph import saved_graph_status
                state = saved_graph_status(project)
                counts, usage, status = state['counts'], state['usage'], state['status']
                failure, stopped = str(error), True
            with graph_for(project) as graph:
                predictions[case['id']] = {'entities': [asdict(row) for row in graph.entities()],
                                           'edges': [asdict(row) for row in graph.edges()], 'failure': failure}
            runs.append({'case': case['id'], 'seconds': time.perf_counter() - started,
                         'counts': counts, 'usage': usage, 'status': status})
            if recorder is not None:
                records.extend(recorder.records[offset:])
            result = score_extractions(corpus, predictions, review=review, execution=execution)
            result.update(runs=runs, predictions=predictions,
                          provider=getattr(provider, 'provider', ''), model=getattr(provider, 'model', ''),
                          scope='Exact annotated identities, relation labels and qualifications. '
                                'Unlisted synonyms require annotation review. Replay scores are not model accuracy.')
            result.update(**extraction_metadata(provider, ExtractionOptions(repair_quotes=False)), usage_by_role={'extraction': usage_summary(records),
                          'answering': usage_summary([]), 'embedding': usage_summary([])})
            if output:
                write_report(output, result)
            if stopped:
                break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--review-template', action='store_true')
    parser.add_argument('--live', action='store_true', help='Call the selected provider; default is annotation replay.')
    parser.add_argument('--extraction-provider', '--provider', dest='provider')
    parser.add_argument('--extraction-model', '--model', dest='model')
    parser.add_argument('--max-cost-usd', type=float)
    args = parser.parse_args()
    corpus = load_corpus(args.corpus)
    if args.review_template:
        write_report(args.output, {'corpus_sha256': corpus_digest(corpus), 'reviewer': '', 'reviewed_at': '',
                                  'approved': False, 'case_ids': [case['id'] for case in corpus['cases']],
                                  'instructions': 'Review identities, relation equivalents, quotes, qualifications and coverage before approval.'})
        return
    if args.live and (not args.provider or not args.model or args.max_cost_usd is None
                      or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0):
        parser.error('--live requires --provider, --model and positive --max-cost-usd')
    review = json.loads(args.review.read_text(encoding='utf8')) if args.review else None
    provider = None
    try:
        if args.live:
            provider = build_generation_provider(args.provider, Settings.from_env(), model=args.model,
                        recorder=UsageRecorder(cost_ceiling_usd=args.max_cost_usd))
        report = run(corpus, generator=provider, output=args.output, review=review,
                     execution='live' if args.live else 'annotation_replay')
        print(json.dumps({'execution': report['execution'], 'review': report['annotation_review']['status'],
                          'summary': report['summary']}, indent=2))
    finally:
        if provider is not None:
            provider.client.close()


if __name__ == '__main__':
    main()
