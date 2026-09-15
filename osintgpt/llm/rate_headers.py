'''Parse provider quota hints without treating errors as retry instructions.'''

import math
import re


def duration(value):
    '''OpenAI reset durations use units such as 1.5s, 200ms or 6m0s.'''
    value = str(value or '').strip()
    parts = re.findall(r'(\d+(?:\.\d+)?)(ms|s|m|h)', value)
    if not parts or ''.join(number + unit for number, unit in parts) != value:
        return None
    result = sum(float(number) * {'ms': .001, 's': 1, 'm': 60, 'h': 3600}[unit]
                 for number, unit in parts)
    return result if math.isfinite(result) else None


def quota_headers(headers, now):
    headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
    limits, remaining = {}, {}
    for axis in ('requests', 'tokens', 'project-tokens'):
        prefix = 'x-ratelimit-'
        try:
            limit = int(headers.get(prefix + 'limit-' + axis, ''))
            if limit > 0:
                limits[axis] = limit
        except ValueError:
            pass
        try:
            left = int(headers.get(prefix + 'remaining-' + axis, ''))
            reset = duration(headers.get(prefix + 'reset-' + axis))
            if left >= 0 and reset is not None:
                remaining[axis] = {'left': left, 'until': now + reset}
        except ValueError:
            pass
    return limits, remaining
