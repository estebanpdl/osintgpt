'''Provider error classification and retry hints without SDK/network activity.'''

from datetime import datetime, timezone
from email.utils import format_datetime
import ssl

import httpx
import pytest
from google.genai.errors import ClientError, ServerError
from openai import APIConnectionError, APIStatusError

from osintgpt.llm.embedding_errors import classify, retry_delay


def api_error(provider='openai', status=429, payload=None, headers=None):
    response = httpx.Response(status, headers=headers, request=httpx.Request('POST', 'https://offline.invalid'))
    payload = payload if payload is not None else {'message': 'temporary failure'}
    if provider == 'gemini':
        cls = ServerError if status >= 500 else ClientError
        return cls(status, {'error': payload}, response)
    return APIStatusError('temporary failure', response=response, body=payload)


@pytest.mark.parametrize('provider', ['openai', 'gemini', 'voyage'])
@pytest.mark.parametrize('status', [408, 409, 429, 500, 502, 503, 504])
def test_transient_provider_statuses(provider, status):
    assert classify(api_error(provider, status), provider, 1000).retryable


@pytest.mark.parametrize('provider', ['openai', 'gemini', 'voyage'])
@pytest.mark.parametrize('status', [400, 401, 402, 403, 404, 413, 422, 501])
def test_invalid_or_unsupported_requests_are_terminal(provider, status):
    assert not classify(api_error(provider, status), provider, 1000).retryable


@pytest.mark.parametrize('code', [
    'insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit_reached',
    'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
    'organization_usage_limit_exceeded',
])
@pytest.mark.parametrize('provider', ['openai', 'voyage'])
def test_billing_and_usage_errors_override_429_and_retry_hints(code, provider):
    error = api_error(provider, payload={'code': code}, headers={'retry-after': '1'})
    failure = classify(error, provider, 1000)
    assert not failure.retryable and failure.delay == 0


def google_details(*items):
    return {'status': 'RESOURCE_EXHAUSTED', 'details': list(items)}


@pytest.mark.parametrize('violation', [
    {'quotaId': 'EmbedContentRequestsPerDayPerProject'},
    {'quotaMetric': 'embed_content_input_tokens_per_day'},
    {'quotaId': 'EmbedContentRequestsPerMinute', 'quotaValue': '0'},
])
def test_gemini_daily_or_zero_quota_is_terminal_even_with_retry_info(violation):
    payload = google_details(
        {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [violation]},
        {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '2s'},
    )
    assert not classify(api_error('gemini', payload=payload), 'gemini', 1000).retryable


def test_gemini_minute_quota_with_generic_billing_help_is_retryable():
    payload = google_details({'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
        {'quotaId': 'EmbedContentInputTokensPerMinute', 'quotaValue': '10000'}
    ]})
    payload['message'] = 'You exceeded your current quota, please check your plan and billing details.'
    assert classify(api_error('gemini', payload=payload), 'gemini', 1000).retryable


@pytest.mark.parametrize('message', ['Rate limit reached on requests per day (RPD)', 'Daily quota exceeded'])
def test_daily_limit_message_is_terminal(message):
    assert not classify(api_error(payload={'message': message}), 'openai', 1000).retryable


@pytest.mark.parametrize('headers,payload,expected', [
    ({'Retry-After': '1.589'}, {}, 1.589),
    ({'retry-after-ms': '1589'}, {}, 1.589),
    ({'retry-after-ms': '1589', 'Retry-After': '5'}, {}, 5),
    ({}, {'message': 'Please try again in 1.589s. Visit the limits page.'}, 1.589),
    ({}, {'message': 'Please retry in 33.470052706s.'}, 33.470052706),
    ({}, google_details({'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '3.5s'}), 3.5),
    ({}, google_details({'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': {'seconds': '3', 'nanos': 500000000}}), 3.5),
    ({'retry-after': '2'}, google_details({'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '4s'}), 4),
    ({'retry-after': 'nan', 'retry-after-ms': '-1'}, {'message': 'retry in -3s'}, 0),
    ({'retry-after': 'inf'}, {'details': [None, 'bad']}, 0),
    ({}, google_details({'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': 'oops'}), 0),
    ({}, {'details': 'bad'}, 0),
    ({'retry-after': '4000'}, {}, 4000),
])
def test_wait_hints_keep_the_largest_valid_minimum(headers, payload, expected):
    assert retry_delay(headers, payload, 1000) == pytest.approx(expected)


def test_http_date_handles_client_clock_skew():
    headers = {name: format_datetime(datetime.fromtimestamp(value, timezone.utc), usegmt=True)
               for name, value in [('Date', 1000), ('Retry-After', 1010)]}
    assert retry_delay(headers, {}, 1000) == 10
    assert retry_delay(headers, {}, 1020) == 10


def test_network_errors_are_sdk_specific_and_do_not_swallow_local_failures():
    assert classify(APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid')), 'openai', 1000).retryable
    assert classify(httpx.ReadTimeout('temporary'), 'gemini', 1000).retryable
    for error in [OSError('disk unavailable'), RuntimeError('429'), ValueError('invalid response')]:
        assert not classify(error, 'openai', 1000).retryable
        assert not classify(error, 'gemini', 1000).retryable
    assert not classify(api_error(), 'ollama', 1000).retryable


def test_unstructured_error_body_does_not_break_classification():
    assert classify(api_error(payload='server unavailable'), 'openai', 1000).retryable


@pytest.mark.parametrize('cause', [httpx.UnsupportedProtocol('bad URL'), httpx.LocalProtocolError('invalid header'), ssl.SSLCertVerificationError('bad certificate')])
def test_connection_configuration_errors_are_terminal(cause):
    wrapped = APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
    wrapped.__cause__ = cause
    assert not classify(wrapped, 'openai', 1000).retryable
    assert not classify(cause, 'gemini', 1000).retryable
    wrapped = httpx.ConnectError('connection failed')
    wrapped.__cause__ = cause
    assert not classify(wrapped, 'gemini', 1000).retryable
