'''Durable, progress-driven identity conversations with provider-valid restarts.'''

from dataclasses import asdict
import json

from .calling import generation_provider, request_schema
from .checkpoints import SavedInvalidResponse
from .contract import _object
from .grounding import parse_response
from .identity_tools import IdentityTools, RESOLVING_SCHEMA, TOOLS, SUBMIT
from .recipes import fingerprint
from .request_context import ProviderCapacityError, context_failure
from .windowing import encode
from osintgpt.llm.generation import GenerationResponse


SAVED = {'name': 'read_saved_lookup', 'description':
         'Read a durable earlier lookup after a conversation restart or deferred result. '
         'index identifies the saved lookup; offset and limit page its exact JSON text by characters. '
         'Use next_offset until complete, then interpret the original JSON; never treat a partial page as absent evidence.',
         'parameters': _object({'index': {'type': 'integer'}, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}})}


class LookupState:
    def __init__(self, tools):
        self.tools = tools
        self.entries, self.cache, self.reads, self.delivered = [], {}, set(), {}
        self.received = set()
        self.delivery_chars = None

    def read(self, args):
        index, offset, limit = args.get('index'), args.get('offset', 0), args.get('limit')
        if (type(index) is not int or not 0 <= index < len(self.entries) or type(offset) is not int or offset < 0
                or (limit is not None and (type(limit) is not int or limit < 1))):
            return {'status': 'invalid_arguments', 'problem': 'Use an existing index, nonnegative offset and positive limit.'}
        raw = encode(self.entries[index])
        end = len(raw) if limit is None else min(len(raw), offset + limit)
        if offset > len(raw):
            return {'status': 'invalid_arguments', 'problem': 'Offset exceeds the saved lookup.'}
        return {'status': 'ok', 'index': index, 'text': raw[offset:end], 'next_offset': end,
                'complete': end == len(raw), 'total_characters': len(raw)}

    def run(self, name, args):
        key = fingerprint({'name': name, 'arguments': args})
        if name == SAVED['name']:
            result = self.read(args)
            fresh = key not in self.reads
            self.reads.add(key)
            return result, fresh and result.get('status') == 'ok', args.get('index', -1)
        if key in self.cache:
            index = self.cache[key]
            return self.entries[index]['result'], False, index
        evidence = {k: set(v) for k, v in self.tools.evidence.items()}
        result = self.tools.run(name, args)
        self.tools.evidence = evidence  # Only delivered evidence authorizes contextual reuse.
        index = len(self.entries)
        self.entries.append({'name': name, 'arguments': args, 'result': result})
        self.cache[key] = index
        return result, result.get('status') == 'ok', index

    def checkpoint(self):
        return {'entries': self.entries, 'cache': self.cache, 'reads': sorted(self.reads),
                'delivered': self.delivered,
                'received': sorted(self.received),
                'delivery_chars': self.delivery_chars,
                'allowed': sorted(self.tools.allowed), 'evidence': {k: sorted(v) for k, v in self.tools.evidence.items()}}

    def restore(self, saved):
        self.entries, self.cache, self.reads = saved['entries'], saved['cache'], set(saved['reads'])
        self.delivered = saved.get('delivered', {})
        self.received = set(saved.get('received', []))
        self.delivery_chars = saved.get('delivery_chars')
        self.tools.allowed.update(saved['allowed'])
        self.tools.evidence = {k: set(v) for k, v in saved['evidence'].items()}

    def acknowledge(self, result, index):
        if result.get('status') != 'ok':
            return
        self.received.add(fingerprint(result))
        if 'text' in result and 'index' in result:
            index = result['index']
            ranges = self.delivered.setdefault(str(index), [])
            ranges.append([result['next_offset'] - len(result['text']), result['next_offset']])
            covered = 0
            for start, end in sorted(ranges):
                if start > covered:
                    return
                covered = max(covered, end)
            if covered < result['total_characters']:
                return
            result = self.entries[index]['result']
        if result.get('entity_id'):
            self.tools.evidence.setdefault(result['entity_id'], set()).update(e['quote'] for e in result.get('evidence', []))


def _restarted_payload(payload, state, problem=''):
    return {**payload, 'lookup_state': {'saved_results': len(state.entries),
            'instruction': 'Earlier lookup work is saved. Use read_saved_lookup by index to recover exact results, '
                           'or search/read candidates again. Finish the structured extraction when identities are resolved '
                           'or genuinely ambiguous. Do not invent evidence or repeat searches without a reason.',
            'latest_index': len(state.entries) - 1}, **({'completion_problem': problem} if problem else {})}


def _restart(calls, system, payload, state, contract, available, history, problem=''):
    current = _restarted_payload(payload, state, problem)
    supplied = []
    # A new conversation can receive exact tool data as source data without
    # replaying the oversized opaque provider state that preceded that data.
    for exchange in reversed(history):
        for raw in exchange['results'].values():
            result = json.loads(raw)
            candidate = {**current, 'saved_lookup_results': [*supplied, result]}
            if calls.context.fits(system, encode(candidate), contract, available):
                supplied.append(result)
                current = candidate
    return current, supplied


def _delivery(calls, system, payload, contract, available, result, index, state):
    '''Full results are retained; page only delivery when actual capacity requires it.'''
    if state.delivery_chars and result.get('text') is not None and 'index' in result:
        result = state.read({'index': result['index'], 'offset': result['next_offset'] - len(result['text']),
                             'limit': min(len(result['text']), state.delivery_chars)})
    if calls.context.fits(system, encode({**payload, 'lookup_result': result}), contract, available):
        return result
    if result.get('text') is not None and result.get('index') is not None:
        text = result['text']
        while text:
            text = text[:len(text) // 2]
            page = {**result, 'text': text, 'next_offset': result['next_offset'] - len(result['text']) + len(text), 'complete': False}
            if text and calls.context.fits(system, encode({**payload, 'lookup_result': page}), contract, available):
                return page
    deferred = {'status': 'deferred', 'saved_index': index,
                'note': 'Result is preserved. Read with read_saved_lookup and follow next_offset; this is not a no-match result.'}
    if not calls.context.fits(system, encode({**payload, 'lookup_result': deferred}), contract, available):
        raise ProviderCapacityError('The provider cannot fit source instructions and a minimal lookup reference. Saved work is preserved.')
    return deferred


def _reduce_rejected_delivery(current, state):
    '''Learn smaller pages from an actual provider rejection, without inventing model capacity.'''
    supplied = current.get('saved_lookup_results', [])
    if len(supplied) > 1:
        supplied = supplied[:max(1, len(supplied) // 2)]
    elif supplied:
        result = supplied[0]
        if 'text' not in result or 'index' not in result:
            index = next((i for i, entry in enumerate(state.entries) if entry['result'] == result), None)
            if index is None:
                return None
            result = state.read({'index': index})
        if len(result['text']) <= 1:
            return None
        size = len(result['text']) // 2
        state.delivery_chars = min(state.delivery_chars or size, size)
        supplied = [state.read({'index': result['index'], 'offset': result['next_offset'] - len(result['text']), 'limit': size})]
    else:
        return None
    return {**current, 'saved_lookup_results': supplied}, supplied


def extraction_system(system, submission):
    return (system + '\nReturn the final extraction by calling submit_extraction once, alone. Do not return raw JSON text.'
            if submission else system)


def extract_with_lookup(calls, system, payload, registry):
    tools = IdentityTools(registry)
    tools.allowed.update(item['id'] for item in payload.get('known_entities', []))
    state = LookupState(tools)
    submission = getattr(generation_provider(calls.generator), 'graph_submission_tool', False)
    contract = None if submission else RESOLVING_SCHEMA
    system = extraction_system(system, submission)
    history, seeded, turn, finalizing, correction = [], [], 0, False, ''
    current = payload
    seen_failures = set()
    while True:
        available = ([SAVED] if finalizing and state.entries else [] if finalizing else [*TOOLS, SAVED]) + ([SUBMIT] if submission else [])
        completion = ('Finish the complete structured extraction using supported evidence; leave genuinely uncertain identities separate. '
                      + correction) if finalizing else correction
        purpose = 'extraction' if turn == 0 else f'identity-turn-{turn}'
        saved = calls.has_saved_response(system, encode(current), contract, purpose=purpose,
                                         tools=available, history=history)
        if not saved and calls.context.should_restart(system, encode(current), contract, available, history):
            current, seeded = _restart(calls, system, payload, state, contract, available, history, completion)
            history = []
            calls.counts.context_restarts += 1
            calls.emit('reorganizing_context')
        if not saved:
            calls.context.check(system, encode(current), contract, available, history)
        calls.emit('completing_extraction' if finalizing or correction else 'finding_identities')
        try:
            response = calls.generate(system, encode(current), contract, purpose=purpose,
                                      tools=available, history=history, return_response=True,
                                      stage='structured_correction' if correction else 'completion' if finalizing else 'extraction')
        except SavedInvalidResponse as error:
            response = error.response  # inspected below; it is never accepted without validation
        except Exception as error:
            if context_failure(error) and history:
                # A provider rejection is new evidence of capacity, not a transient retry.
                current, seeded = _restart(calls, system, payload, state, contract, available, history, completion)
                history = []
                calls.counts.context_restarts += 1
                calls.emit('reorganizing_context')
                continue
            if context_failure(error):
                smaller = _reduce_rejected_delivery(current, state)
                if smaller:
                    current, seeded = smaller
                    calls.counts.context_restarts += 1
                    calls.emit('reorganizing_context')
                    continue
            raise
        for result in seeded:
            state.acknowledge(result, -1)
        seeded = []
        for exchange in history:
            for raw in exchange['results'].values():
                state.acknowledge(json.loads(raw), -1)
        submitted = [c for c in response.tool_calls if c['name'] == SUBMIT['name']]
        if not response.tool_calls or submitted:
            try:
                response.require_complete(allow_tools=True)
                if submission and (len(submitted) != 1 or len(response.tool_calls) != 1):
                    raise ValueError('Expected submit_extraction as the only final tool call.')
                reply = encode(submitted[0]['arguments']) if submitted else response.text
                parse_response(reply, schema=request_schema(calls.generator, RESOLVING_SCHEMA))
                calls.validated()
                return reply, tools
            except ValueError as error:
                calls.invalidate(str(error))
                calls.counts.invalid_responses += 1
                # Repeated structural failure is non-progress, independent of response length.
                if correction:
                    raise ValueError('Structured completion made no progress; original and correction responses are saved. '
                                     + str(error)) from error
                correction = str(error)
                calls.counts.structured_corrections += 1
                finalizing = True
                current, seeded = _restart(calls, system, payload, state, contract, available, history,
                    'Return a complete structured extraction from the original source. ' + correction)
                history = []
                turn += 1
                continue
        if finalizing and any(c['name'] != SAVED['name'] for c in response.tool_calls):
            calls.invalidate('Model requested more lookup work after explicit non-progress finalization.')
            raise ValueError('Identity lookup made no progress; provider did not complete structured extraction. Responses are saved.')
        key = fingerprint({'calls': response.tool_calls, 'catalogue': tools.catalog.version, 'policy': 2,
                           'state': fingerprint(state.checkpoint())})
        saved = calls.checkpoint.response(calls.position, f'lookup-{turn}', key) if calls.checkpoint else None
        if saved:
            replay = json.loads(saved.text)
            state.restore(replay['state'])
            results, progress = replay['results'], replay['progress']
            calls.counts.entity_lookups_reused += len(response.tool_calls)
        else:
            results, progress, failures = {}, False, set()
            calls.emit('retrieving_evidence')
            for call in response.tool_calls:
                result, fresh, index = state.run(call['name'], call['arguments'])
                calls.counts.entity_lookups += int(fresh)
                calls.counts.entity_lookups_reused += int(not fresh)
                progress |= fresh
                if result.get('status') == 'invalid_arguments':
                    failures.add(result['problem'])
                result = _delivery(calls, system, payload, contract, available, result, index, state)
                progress |= result.get('status') == 'ok' and fingerprint(result) not in state.received
                results[call['id']] = encode(result)
            # One chance to correct a new argument error; repeated errors are non-progress.
            progress |= bool(failures - seen_failures)
            seen_failures.update(failures)
            if calls.checkpoint:
                calls.checkpoint.save_response(calls.position, f'lookup-{turn}', key,
                    GenerationResponse(encode({'state': state.checkpoint(), 'results': results, 'progress': progress})))
        calls.validated()
        history.append({'response': asdict(response), 'results': results})
        if not progress:
            if finalizing:
                calls.invalidate('Repeated saved-evidence reads made no progress during finalization.')
                raise ValueError('Structured completion made no progress; repeated lookup responses are saved.')
            finalizing = True
            calls.counts.no_progress_stops += 1
            current = {**current, 'completion_instruction': 'These lookups repeated completed work without progress. '
                       'Finish the structured extraction using supported evidence; leave genuinely uncertain identities separate.'}
        turn += 1
        calls.notify()
