'''Paired answers with independent model roles and frozen evidence; offline transport is the default.'''

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

from osintgpt.config import Settings
from osintgpt.graph.answer_evaluation import citation_bounds, review_template, score_answer_reviews
from osintgpt.llm import build_embedding_provider, build_generation_provider
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from graph_answer_evidence import prepare, validate_bundle
from graph_benchmark import RoleUsage, model_roles, usage_summary
from graph_fixtures import write_report


def run(fixture, generator, embedder, recorder, *, extractor=None, evidence=None,
        execution='injected_providers', budget=2048, output_tokens=1024, output=None):
    if type(output_tokens) is not int or output_tokens <= 0:
        raise ValueError('Output token budget must be a positive integer.')
    extractor = (extractor if extractor is not None else generator) if evidence is None else None
    if evidence is None and embedder is None:
        raise ValueError('An embedder is required to prepare new evidence.')
    usage = RoleUsage(recorder, answering=generator, extraction=extractor, embedding=embedder)
    offset = len(recorder)
    answers = []
    report = {'status': 'running', 'execution': execution, 'scope': 'Controlled supplied-context comparison. '
              'Fixed retrieval requests and equal rendered input/output limits; autonomous planning is not evaluated. '
              'Offline scripts and simulated usage do not measure model quality or provider spend.',
              'models': model_roles(generator, extractor, embedder), 'evidence_reused': evidence is not None,
              'context_token_budget': budget, 'output_token_budget': output_tokens, 'answers': answers,
              'encoding': 'cl100k_base', 'review_status': 'pending', 'cost_ceiling_usd': recorder.cost_ceiling_usd}
    if execution == 'offline_transport':
        report['actual_provider_spend_usd'] = 0
    started = time.perf_counter()
    try:
        bundle = deepcopy(validate_bundle(evidence, fixture, budget, output_tokens)) if evidence is not None else prepare(
            fixture, extractor, embedder, usage, budget, report, output_tokens)
        report.update(evidence_bundle=bundle, evidence_sha256=bundle['sha256'], graph_build=bundle['graph_build'],
                      comparison_controls=bundle['controls'])
        if evidence is not None:
            report['models'].update(extraction=bundle['models']['extraction'], embedding=bundle['models']['embedding'],
                                    extraction_selection='frozen_evidence')
        if output:
            write_report(output, report)
        for item in bundle['rows']:
            row = deepcopy(item)
            row['answering_model'] = report['models']['answering']
            before, call_start = len(recorder), time.perf_counter()
            with usage.stage('answering'):
                row['answer'] = generator.generate(row['system_input'], row['question'], max_output_tokens=output_tokens
                    ) if row['context']['passages'] else 'No evidence retrieved.'
            summary = usage_summary(recorder.records[before:])
            row.update(seconds=time.perf_counter() - call_start, usage=summary, tokens=summary['tokens'],
                       estimated_cost_usd=summary['estimated_cost_usd'], output_token_budget=output_tokens,
                       citations=citation_bounds(row['answer'], row['context']['passages']))
            answers.append(row)
            if output:
                write_report(output, report)
        report['status'] = 'complete_pending_review'
    except CostLimitReached as error:
        report.update(status='cost_stopped', problem=str(error))
    except Exception as error:
        report.update(status='failed', problem=str(error))
    finally:
        summary = usage_summary(recorder.records[offset:])
        report.update(seconds=time.perf_counter() - started, calls=summary['calls'], tokens=summary['tokens'],
                      estimated_cost_usd=summary['estimated_cost_usd'], cost_complete=summary['cost_complete'],
                      usage=summary, budget_usage=usage_summary(recorder.records), usage_by_role=usage.report(),
                      review_template=review_template(answers))
        if output:
            write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path(__file__).with_name('graph_passages.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Explicitly enable provider calls. Default uses local HTTP fixtures.')
    parser.add_argument('--answer-provider', '--provider', dest='answer_provider')
    parser.add_argument('--answer-model', '--model', dest='answer_model')
    parser.add_argument('--extraction-provider')
    parser.add_argument('--extraction-model')
    parser.add_argument('--embedding-provider')
    parser.add_argument('--embedding-model')
    parser.add_argument('--max-cost-usd', type=float)
    parser.add_argument('--reuse-evidence', type=Path, help='Reuse a prior report’s frozen graph and contexts; only answers run.')
    parser.add_argument('--context-tokens', type=int, default=2048)
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--score-run', type=Path)
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    if args.score_run:
        if not args.review:
            parser.error('--score-run requires --review')
        report = json.loads(args.score_run.read_text(encoding='utf8'))
        if report['status'] != 'complete_pending_review':
            parser.error('Only complete paired runs can be scored.')
        write_report(args.output, score_answer_reviews(report['answers'], json.loads(args.review.read_text(encoding='utf8'))))
        return
    if min(args.context_tokens, args.output_tokens) <= 0:
        parser.error('Context and output token budgets must be positive.')
    fixture = json.loads(args.fixture.read_text(encoding='utf8'))
    evidence = None
    if args.reuse_evidence:
        if any((args.extraction_provider, args.extraction_model, args.embedding_provider, args.embedding_model)):
            parser.error('--reuse-evidence cannot be combined with extraction or embedding selections.')
        try:
            evidence = validate_bundle(json.loads(args.reuse_evidence.read_text(encoding='utf8')).get('evidence_bundle'),
                                       fixture, args.context_tokens, args.output_tokens)
        except ValueError as error:
            parser.error(str(error))
    if args.live:
        if (not args.answer_provider or not args.answer_model or args.max_cost_usd is None
                or not math.isfinite(args.max_cost_usd) or args.max_cost_usd <= 0):
            parser.error('--live requires answering provider/model and a finite positive max-cost-usd.')
        if evidence is None and not all((args.embedding_provider, args.embedding_model)):
            parser.error('New evidence requires embedding-provider and embedding-model.')
        if args.extraction_provider and not args.extraction_model:
            parser.error('An extraction-provider override requires extraction-model; it never borrows the answering model.')
    elif any((args.answer_provider, args.answer_model, args.extraction_provider, args.extraction_model,
              args.embedding_provider, args.embedding_model, args.max_cost_usd is not None)):
        parser.error('Provider/model and spending options require explicit --live; default execution is offline.')
    recorder = UsageRecorder(cost_ceiling_usd=args.max_cost_usd)
    answerer = extractor = embedder = None
    try:
        if args.live:
            settings = Settings.from_env()
            answerer = build_generation_provider(args.answer_provider, settings, model=args.answer_model, recorder=recorder)
            if evidence is None:
                extractor = (build_generation_provider(args.extraction_provider or args.answer_provider, settings,
                             model=args.extraction_model, recorder=recorder) if args.extraction_model else answerer)
                embedder = build_embedding_provider(args.embedding_provider, settings, model=args.embedding_model, recorder=recorder)
        else:
            from graph_offline import RecordedFixtureEmbedder, generation
            answerer = generation(fixture, recorder, role='answering', model='offline-answerer')
            if evidence is None:
                extractor = generation(fixture, recorder, role='extraction', model='offline-extractor')
                embedder = RecordedFixtureEmbedder(recorder)
        report = run(fixture, answerer, embedder, recorder, extractor=extractor, evidence=evidence,
                     execution='live' if args.live else 'offline_transport', budget=args.context_tokens,
                     output_tokens=args.output_tokens, output=args.output)
        print(json.dumps({key: report[key] for key in ('status', 'execution', 'calls', 'tokens', 'estimated_cost_usd')}))
        if report['status'] != 'complete_pending_review':
            raise SystemExit(1)
    finally:
        closed = set()
        for provider in (answerer, extractor, embedder):
            if provider is not None and id(provider) not in closed and hasattr(provider, 'client'):
                provider.client.close()
                closed.add(id(provider))


if __name__ == '__main__':
    main()
