'''Classify paid embedding failures and read minimum provider retry delays.'''

import math
import re
import ssl
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx
from google.genai.errors import APIError as GeminiAPIError
from openai import APIConnectionError, APIStatusError

from .rate_headers import duration

TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
ACTION_REQUIRED = {
    'insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit_reached',
    'billing_not_active', 'billing_disabled', 'billing_not_enabled',
    'organization_spend_limit_exceeded', 'project_spend_limit_exceeded',
    'organization_usage_limit_exceeded', 'daily_quota_exceeded',
}
TRANSIENT_TRANSPORT = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError, httpx.ProxyError)


@dataclass(frozen=True)
class Failure:
    retryable: bool
    status: int | None = None
    delay: float = 0.0


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def http_date(value):
    try:
        parsed = parsedate_to_datetime(value)
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (TypeError, ValueError, OverflowError):
        return None


def error_headers(error):
    return getattr(getattr(error, 'response', None), 'headers', None) or {}


def retry_delay(headers, payload, now):
    '''Use the largest valid minimum; never shorten a provider's wait hint.'''
    headers = {str(key).lower(): str(value) for key, value in headers.items()}
    waits = [0.0]
    milliseconds = number(headers.get('retry-after-ms'))
    if milliseconds is not None:
        waits.append(milliseconds / 1000)
    value = headers.get('retry-after')
    seconds = number(value)
    if seconds is not None:
        waits.append(seconds)
    else:
        date = http_date(value)
        if date is not None:
            # Date also protects against a client clock running ahead.
            server_now = http_date(headers.get('date'))
            waits.append(max(0, date - now, date - server_now if server_now is not None else 0))
    details = payload.get('details', [])
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict) or detail.get('@type') != 'type.googleapis.com/google.rpc.RetryInfo':
            continue
        value = detail.get('retryDelay')
        if isinstance(value, dict):
            seconds, nanos = number(value.get('seconds', 0)), number(value.get('nanos', 0))
            parsed = seconds + nanos / 1e9 if seconds is not None and nanos is not None and nanos < 1e9 else None
        else:
            parsed = duration(value)
        if parsed is not None:
            waits.append(parsed)
    # Used by OpenAI and Gemini when a precise delay is only in the message.
    message = payload.get('message', '')
    if isinstance(message, str):
        for match in re.finditer(r'\b(?:try again|retry) in (\d+(?:\.\d+)?(?:ms|s|m|h))\b', message, re.I):
            parsed = duration(match.group(1).lower())
            if parsed is not None:
                waits.append(parsed)
    return max(waits)


def requires_action(payload):
    for field in ('code', 'type', 'status'):
        if str(payload.get(field, '')).lower() in ACTION_REQUIRED:
            return True
    details = payload.get('details', [])
    for detail in details if isinstance(details, list) else []:
        if not isinstance(detail, dict):
            continue
        if str(detail.get('reason', '')).lower() in ACTION_REQUIRED:
            return True
        if detail.get('@type') != 'type.googleapis.com/google.rpc.QuotaFailure':
            continue
        violations = detail.get('violations', [])
        for violation in violations if isinstance(violations, list) else []:
            if not isinstance(violation, dict):
                continue
            quota = ' '.join(str(violation.get(key, '')) for key in ('quotaId', 'quotaMetric')).lower()
            if any(marker in quota for marker in ('perday', 'per_day', 'daily', 'permonth', 'per_month')):
                return True
            if number(violation.get('quotaValue')) == 0:
                return True
    message = str(payload.get('message', '')).lower()
    return bool(re.search(
        r'\b(?:requests|tokens) per day\b|\b(?:daily|monthly) (?:quota|limit)\b|'
        r'\b(?:insufficient|exhausted) (?:credits|credit balance)\b', message
    ))


def connection_requires_action(error):
    # SDKs wrap transport errors. Invalid URLs/protocols and certificate
    # verification failures need configuration changes, not another request.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, (httpx.UnsupportedProtocol, httpx.LocalProtocolError, ssl.SSLCertVerificationError)):
            return True
        error = error.__cause__
    return False


def classify(error, provider, now):
    '''Unknown/local/programming errors are terminal, regardless of their text.'''
    if provider in ('openai', 'voyage'):
        if isinstance(error, APIConnectionError):
            return Failure(not connection_requires_action(error))
        if not isinstance(error, APIStatusError):
            return Failure(False)
        status, payload = error.status_code, error.body
    elif provider == 'gemini':
        if isinstance(error, TRANSIENT_TRANSPORT):
            return Failure(not connection_requires_action(error))
        if not isinstance(error, GeminiAPIError):
            return Failure(False)
        status, payload = error.code, error.details
    else:
        return Failure(False)
    if not isinstance(payload, dict):
        payload = {}
    if isinstance(payload.get('error'), dict):
        payload = payload['error']
    retryable = status in TRANSIENT_STATUS and not requires_action(payload)
    return Failure(retryable, status, retry_delay(error_headers(error), payload, now) if retryable else 0)
