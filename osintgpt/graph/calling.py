'''Durable responses, bounded HTTP attempts, and extraction accounting.'''

from inspect import signature
import json

from osintgpt.llm.generation import GenerationResponse
from osintgpt.llm.generation_pacing import GenerationRequests

from .events import check_cancelled, emit_graph
from .recipes import fingerprint
from .contract import legacy_field_request, legacy_repair_payload, normalize_saved_extraction
from .output_policy import policy_for, policy_data


def request_schema(generator, schema):
    if (getattr(generator, 'supports_structured_output', False)
            and callable(getattr(generator, 'generate_response', None))):
        return schema
    return None


def generation_provider(generator):
    return getattr(generator, 'generation_provider', generator)


def response_hooks(generator):
    method = getattr(generation_provider(generator), 'generate_response', None)
    return signature(method).parameters if callable(method) else {}


def bounded_generate(generator, system, user, output_tokens, schema=None, on_response=None, request=None,
                     graph_output=None, tools=None, history=None):
    response_method = getattr(generator, 'generate_response', None)
    if callable(response_method):
        hooks = response_hooks(generator)
        options = {key: value for key, value in {'on_response': on_response, 'request': request,
                                              'graph_output': graph_output, 'tools': tools, 'history': history}.items()
                   if key in hooks and value is not None}
        result = response_method(system, user, max_output_tokens=output_tokens, schema=schema, **options)
        if on_response is not None and 'on_response' not in options:
            on_response(result)
        return result
    options = {'max_output_tokens': output_tokens} if supports_output_limit(generator) else {}
    result = GenerationResponse(generator.generate(system, user, **options))
    if on_response is not None:
        on_response(result)
    return result


def supports_output_limit(generator):
    supported = getattr(generator, 'supports_output_limit', None)
    if supported is not None:
        return supported
    method = getattr(generator, 'generate_response', None) or generator.generate
    return 'max_output_tokens' in signature(method).parameters


class ExtractionCalls:
    def __init__(self, generator, options, counts, progress, checkpoint=None, requests=None):
        self.generator, self.options = generator, options
        self.counts, self.progress = counts, progress
        self.checkpoint = checkpoint
        self.requests = requests or GenerationRequests()
        self.position, self.purpose = 0, 'extraction'
        self.stage = 'extraction'
        self.policy = policy_for(generator)
        from .request_context import RequestContext
        self.context = RequestContext(generation_provider(generator), options)

    def notify(self):
        if self.progress:
            self.progress(self.counts)

    def emit(self, kind, **data):
        if kind == 'retry':
            self.counts.retries += 1
            self.notify()
        emit_graph(kind, ref=self.checkpoint.ref if self.checkpoint else '',
                   window=self.position, purpose=self.purpose, **data)

    def fits(self, system, user, schema=None, *, purpose=None):
        if purpose and self.has_saved_response(system, user, schema, purpose=purpose):
            return True
        return self.context.fits(system, user, request_schema(self.generator, schema))

    def invalidate(self, problem='Invalid response.'):
        if self.checkpoint:
            self.checkpoint.discard_response(self.position, self.purpose, problem)

    def validated(self):
        if self.checkpoint:
            self.checkpoint.discard_response(self.position, self.purpose, '', state='validated')

    def _ready(self):
        check_cancelled()

    def _attempt(self):
        self._ready()
        self.counts.requests_attempted += 1
        self.counts.completion_requests += int(self.stage == 'completion')
        self.counts.structured_correction_requests += int(self.stage == 'structured_correction')
        self.notify()

    def _failed(self, error):
        if isinstance(error, Exception):
            self.counts.requests_failed += 1
        else:
            self.counts.requests_interrupted += 1
        self.notify()

    def _saved_response(self, system, user, schema, purpose, tools, history):
        """Exact request lookup, including the reviewed previous field spelling."""
        native_schema = request_schema(self.generator, schema)
        key = fingerprint({'request': self.context.arguments(system, user, native_schema, tools, history),
                           'output_policy': policy_data(self.generator)})
        reply = self.checkpoint.response(self.position, purpose, key) if self.checkpoint else None
        recovered = False
        if reply is None and self.checkpoint:
            old_system, old_schema, old_tools = legacy_field_request(system, native_schema, tools)
            old_user = legacy_repair_payload(user, purpose)
            old_key = fingerprint({'request': self.context.arguments(old_system, old_user, old_schema, old_tools, history),
                                   'output_policy': policy_data(self.generator)})
            if old_key != key:
                reply = self.checkpoint.response(self.position, purpose, old_key)
                recovered = reply is not None
        if reply is None and self.checkpoint and self.checkpoint.legacy:
            reply = self.checkpoint.previous(self.position, purpose)
            recovered = reply is not None
        return reply, recovered, key

    def has_saved_response(self, system, user, schema=None, *, purpose='extraction', tools=None, history=None):
        """Avoid changing a reusable request merely to fit the longer new field name."""
        if not self.checkpoint:
            return False
        from .checkpoints import SavedInvalidResponse
        try:
            return self._saved_response(system, user, schema, purpose, tools, history)[0] is not None
        except SavedInvalidResponse:
            return True  # The normal recovery path must inspect the original failure.

    def generate(self, system, user, schema=None, *, purpose='extraction', tools=None, history=None, return_response=False,
                 stage='extraction'):
        self.purpose = purpose
        self.stage = stage
        check_cancelled()
        native_schema = request_schema(self.generator, schema)
        reply, recovered, key = self._saved_response(system, user, schema, purpose, tools, history)
        if reply is not None:
            if recovered:
                reply = normalize_saved_extraction(reply)
                self.counts.responses_recovered += 1
            else:
                self.counts.responses_reused += 1
            self.emit('response_recovered' if recovered else 'response_reused')
        else:
            self.context.check(system, user, native_schema, tools, history)
            reply = self._generate(system, user, native_schema, key, schema is not None, tools, history)
        try:
            if not reply.has_completion_status:
                self.counts.completion_status_unavailable += 1
            text = reply.require_complete(allow_tools=return_response)
            return reply if return_response else text
        except ValueError as error:
            self.invalidate(str(error))
            self.counts.invalid_responses += 1
            raise
        finally:
            self.notify()

    def _generate(self, system, user, schema, key, structured, tools=None, history=None):
        if self.purpose.startswith('repair'):
            self.counts.repairs_attempted += 1
        hooks = response_hooks(self.generator)
        provider = generation_provider(self.generator)
        controlled = 'request' in hooks and 'on_response' in hooks and hasattr(provider, 'client')
        delivered = False
        recorder = getattr(self.generator, 'recorder', None)
        before = len(recorder) if recorder is not None else 0

        def receive(reply):
            nonlocal delivered
            delivered = True
            self.counts.requests_succeeded += 1
            if self.checkpoint:
                source = json.loads(user).get('source', {})
                self.checkpoint.save_response(self.position, self.purpose, key, reply,
                    {'source_range': {'start': source.get('start'), 'end': source.get('end')}})
                self.counts.responses_saved += 1
            self.notify()
            self.emit('response_saved' if self.checkpoint else 'response_received')

        request = None
        if controlled:
            pacer = self.requests.for_provider(provider)
            tokens = self.context.measure(system, user, schema, tools, history)
            request = lambda operation: pacer.request(operation, self.policy.reservation(tokens),
                before=self._attempt, failed=self._failed, emit=self.emit, check_cancel=self._ready)
            request.check_cancelled = check_cancelled
        else:
            self.counts.request_controls_unenforced += 1
            self._attempt()
        if structured and schema is None:
            self.counts.structured_output_unenforced += 1
        try:
            return bounded_generate(self.generator, system, user, None,
                                    schema, on_response=receive, request=request,
                                    graph_output=self.policy.max_tokens, tools=tools, history=history)
        except BaseException as error:
            if not delivered and not controlled:
                if recorder is not None and len(recorder) > before:
                    self.counts.requests_succeeded += 1
                else:
                    self._failed(error)
            raise
        finally:
            self.notify()
