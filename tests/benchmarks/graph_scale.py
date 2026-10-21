'''Synthetic PDF recovery benchmark using native SDK serialization with an offline HTTP transport.'''

import argparse
import json
import math
import re
import tempfile
import time
import tracemalloc
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from osintgpt import Project
from osintgpt.graph import ExtractionOptions, build_graph, graph_for, saved_graph_status
from osintgpt.graph.events import GraphCancelled
from osintgpt.ingestion import Corpus
from osintgpt.llm.generation_pacing import GenerationLimits, GenerationRequests
from osintgpt.llm.openai_responses import OpenAIResponsesGeneration
from osintgpt.llm.rate_ledger import RateLedger
from osintgpt.llm.usage import CostLimitReached, UsageRecorder
from osintgpt.rate_settings import RateLimits
from graph_fixtures import write_report
from graph_benchmark import extraction_metadata, usage_summary
from graph_offline import with_resolution


def create_pdf(path, pages, *, changed=False):
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')}))
    for number in range(1, pages + 1):
        page = writer.add_blank_page(width=612, height=792)
        target = 'Revised' if changed and number == 1 else 'Organization'
        lines = [f'Page {number}. Entity {number:04d} supports {target} {number:04d}.']
        lines += [f'Record {number}, detail {i}: This fictional administrative appendix describes procedures and schedules.'
                  for i in range(18)]
        contents = DecodedStreamObject()
        commands = ['BT /F1 9 Tf 36 750 Td 13 TL'] + [f'({line}) Tj T*' for line in lines] + ['ET']
        contents.set_data('\n'.join(commands).encode('ascii'))
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        page[NameObject('/Contents')] = writer._add_object(contents)
    with Path(path).open('wb') as stream:
        writer.write(stream)
    writer.close()


def proposal(payload):
    known = {entity['name']: entity['id'] for entity in payload['known_entities']
             if entity['document_ref'] == payload['document_ref'] and entity['source_version'] == payload['source_version']}
    entities, edges = {}, []
    for match in re.finditer(r'(Entity \d{4}) supports ((?:Organization|Revised) \d{4})\.', payload['source']['text']):
        source, target = match.groups()
        for name in (source, target):
            entities[name] = {'id': name, 'name': name, 'type': 'organization', 'description': '',
                              'existing_id': known.get(name, ''), 'evidence': [name], 'aliases': [], 'identifiers': []}
        edges.append({'source': source, 'target': target, 'relation': 'supports', 'evidence': [match.group()],
                      'relation_polarity': 'positive', 'assertion': 'stated', 'attributed_to': '',
                      'event_time': '', 'statement_time': ''})
    return {'entities': list(entities.values()), 'edges': edges}


class Transport:
    def __init__(self):
        self.faults = deque()
        self.attempts = 0

    def handle(self, request):
        self.attempts += 1
        fault = self.faults.popleft() if self.faults else ''
        if fault == '429':
            return httpx.Response(429, headers={'retry-after': '2'}, json={
                'error': {'code': 'rate_limit_exceeded', 'message': 'Injected transient limit'}})
        body = json.loads(request.content)
        payload = json.loads(body['input'][0]['content'])
        result = {'invalid': True} if fault == 'invalid' else with_resolution(proposal(payload), body)
        text = json.dumps(result)
        inputs, outputs = math.ceil(len(request.content) / 4), math.ceil(len(text) / 4)
        return httpx.Response(200, json={'id': f'resp_{self.attempts}', 'object': 'response', 'created_at': 0,
            'model': 'gpt-4o-mini', 'status': 'completed', 'incomplete_details': None,
            'output': [{'id': 'msg', 'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text, 'annotations': []}]}],
            'usage': {'input_tokens': inputs, 'output_tokens': outputs, 'total_tokens': inputs + outputs}})


class Clock:
    def __init__(self):
        self.now, self.waited = 1000.0, 0.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.waited += seconds


def run(pages=1000, output=None):
    if type(pages) is not int or pages < 30:
        raise ValueError('Use at least 30 pages to exercise interrupted windows.')
    transport, clock, stages, events = Transport(), Clock(), [], Counter()
    provider = OpenAIResponsesGeneration('gpt-4o-mini', 'offline-fixture-key', recorder=UsageRecorder())
    provider.client.close()
    provider.client = OpenAI(api_key='offline-fixture-key', http_client=httpx.Client(transport=httpx.MockTransport(transport.handle)))
    result = {'scope': 'Synthetic PDF and scripted extraction; real loaders, windows, native SDK, checkpoints and publication. '
                       'Tokens and catalog cost are simulated usage, not observed model usage; actual provider spend is zero. '
                       'Python allocation measurements exclude native PDF/SDK memory and include tracing overhead.',
              'pages': pages, 'stages': stages, 'actual_provider_spend_usd': 0, 'passed': False}
    tracemalloc.start()
    try:
        with tempfile.TemporaryDirectory(prefix='osintgpt-scale-') as root:
            project = Project.create('Synthetic scale', home=root).with_settings(graph_enabled=True)
            path = project.paths.root / 'appendix.pdf'
            create_pdf(path, pages)
            Corpus.load(project.paths.sources).register(path.name)
            result['source_bytes'] = path.stat().st_size
            ledger = RateLedger(Path(root) / 'rates.sqlite', clock=clock.time, sleep=clock.sleep)
            limits = GenerationLimits(rates=RateLimits(rpm=600))
            options = ExtractionOptions(repair_quotes=False)
            result.update(**extraction_metadata(provider, options), execution='offline_transport')

            def attempt(name, stop_after=None, stop_before_completion=False, rebuild=False):
                started, before, saved = time.perf_counter(), transport.attempts, 0
                def observe(event):
                    nonlocal saved
                    events[event.kind] += 1
                    if stop_before_completion and event.kind == 'completing_extraction':
                        raise GraphCancelled('Injected boundary before correcting a saved invalid response')
                    if event.kind == 'window_saved':
                        saved += 1
                        if stop_after and saved == stop_after:
                            raise GraphCancelled('Injected process boundary after durable window save')
                def requests(value):
                    controller = GenerationRequests(value, ledger=ledger)
                    controller.for_provider(provider).retry_jitter = lambda a, b: 0
                    return controller
                stopped = ''
                with patch('osintgpt.graph.build.GenerationRequests', requests):
                    try:
                        build_graph(project, provider, options=options, request_limits=limits, on_event=observe, rebuild=rebuild)
                    except (GraphCancelled, CostLimitReached) as error:
                        stopped = type(error).__name__
                state = saved_graph_status(project)
                stages.append({'name': name, 'seconds': time.perf_counter() - started,
                    'transport_attempts': transport.attempts - before, 'status': state['status'], 'stopped': stopped,
                    'counts': state['counts'], 'usage': state['usage'], 'checkpoints': state['checkpoints'],
                    'published_edges': state['edges'], 'stale': state['stale'],
                    'peak_python_mib': tracemalloc.get_traced_memory()[1] / 1024 ** 2})
                if output:
                    write_report(output, result)
                return stages[-1]

            transport.faults.append('429')
            # Leave pending work even in the 30-page smoke fixture with the
            # larger input room available after retiring the output reserve.
            interrupted = attempt('rate_limit_then_interrupt', stop_after=2)
            assert interrupted['status'] == 'interrupted' and interrupted['counts']['retries'] == 1
            assert interrupted['checkpoints']['windows_saved'] == 2
            transport.faults.append('invalid')
            invalid = attempt('invalid_response', stop_before_completion=True)
            assert invalid['status'] == 'interrupted' and invalid['counts']['invalid_responses'] == 1
            provider.recorder.cost_ceiling_usd = 0
            stopped = attempt('cost_stop_after_saved_response')
            assert stopped['stopped'] == 'CostLimitReached' and stopped['checkpoints']['responses_saved'] == 2
            provider.recorder.cost_ceiling_usd = None
            resumed = attempt('resume_to_completion')
            assert resumed['status'] == 'complete' and resumed['counts']['responses_reused'] == 1
            assert resumed['counts']['windows_reused'] == 2 and resumed['published_edges'] == pages
            unchanged = attempt('unchanged_reuse')
            assert unchanged['transport_attempts'] == 0
            create_pdf(path, pages, changed=True)
            transport.faults.extend(['invalid', 'invalid'])
            failed = attempt('changed_source_failed_replacement')
            assert failed['status'] == 'complete_with_gaps' and failed['stale'] and failed['published_edges'] == pages
            with graph_for(project) as graph:
                assert any(edge.target == 'Organization 0001' for edge in graph.edges())
            no_repeat = attempt('known_invalid_resume')
            assert no_repeat['transport_attempts'] == 0 and no_repeat['published_edges'] == pages
            updated = attempt('changed_source_recovery', rebuild=True)
            assert updated['status'] == 'complete' and updated['published_edges'] == pages
            with graph_for(project) as graph:
                edges = graph.edges()
                assert any(edge.target == 'Revised 0001' for edge in edges)
                assert not any(edge.target == 'Organization 0001' for edge in edges)
                last = next(edge for edge in edges if edge.source == f'Entity {pages:04d}')
                assert pages in {span['page'] for match in last.provenance['matches'] for span in match['spans']}
            result.update(passed=True, events=dict(events), transport_attempts=transport.attempts,
                          simulated_wait_seconds=clock.waited, simulated_tokens=provider.recorder.total_tokens,
                          simulated_catalog_cost_usd=provider.recorder.estimated_billable_cost,
                          total_stage_seconds=sum(stage['seconds'] for stage in stages),
                          peak_python_mib=tracemalloc.get_traced_memory()[1] / 1024 ** 2)
    finally:
        result['usage_by_role'] = {'extraction': usage_summary(provider.recorder.records),
                                  'answering': usage_summary([]), 'embedding': usage_summary([])}
        tracemalloc.stop()
        provider.client.close()
        if output:
            write_report(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages', type=int, default=1000)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.pages, arguments.output)
    print(json.dumps({key: value for key, value in report.items() if key != 'stages'}, indent=2))
