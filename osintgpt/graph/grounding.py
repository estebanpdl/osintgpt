'''Validate quotations against only the source ranges supplied in a request.'''

from osintgpt.ingestion.provenance import quote_locations
from .contract import validate_structure


class GroundingError(ValueError):
    pass


def quotations(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise GroundingError('Provide at least one evidence quotation.')
    if any(not isinstance(quote, str) or not quote.strip() for quote in value):
        raise GroundingError('Evidence quotations must be nonempty strings.')
    return tuple(dict.fromkeys(quote.strip() for quote in value))


class EvidenceScope:
    def __init__(self, document, snapshot, ranges):
        self.document, self.snapshot, self.ranges = document, snapshot, ranges

    def prove(self, quotes):
        quotes = quotations(quotes)
        supporting = []
        for quote in quotes:
            matches = list({(match['start'], match['end']): match
                            for start, end in self.ranges
                            for match in quote_locations(self.document, quote, after=start, before=end)}.values())
            if not matches:
                raise GroundingError('Quotation not found in supplied source text: ' + quote[:120])
            matches.sort(key=lambda match: match['start'])
            supporting.append({'quote': quote, 'matches': matches})
        provenance = self.snapshot.provenance(self.document, '')
        provenance.update(matches=supporting[0]['matches'], supporting=supporting,
                          match_status='found', supplied_ranges=[list(pair) for pair in self.ranges])
        return quotes, provenance

    def supported(self, quotes):
        '''Independently validate quotations, reporting rejected optional evidence.'''
        kept, warnings = [], []
        for quote in quotations(quotes):
            try:
                self.prove([quote])
                kept.append(quote)
            except GroundingError as error:
                warnings.append(str(error))
        if not kept:
            raise GroundingError('; '.join(warnings))
        valid, provenance = self.prove(kept)
        return valid, provenance, warnings


def parse_response(reply, key='edges', *, schema=None):
    import json
    import re
    if not isinstance(reply, str):
        raise ValueError('Extraction response must be JSON text.')
    if not reply.strip():
        raise ValueError('Empty extraction response; no JSON content was returned.')
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as error:
        fenced = re.fullmatch(r'\s*```(?:json)?\s*([\s\S]*?)\s*```\s*', reply)
        if not fenced:
            raise ValueError(f'Invalid JSON extraction response at line {error.lineno}, column {error.colno}: {error.msg}.') from error
        data = json.loads(fenced[1])
    if schema is not None:
        validate_structure(data, schema)
    keys = ('entities', 'edges') if key == 'edges' else (key,)
    if not isinstance(data, dict) or any(not isinstance(data.get(name), list) for name in keys):
        raise ValueError('Extraction response requires arrays: ' + ', '.join(keys))
    if any(any(not isinstance(row, dict) for row in data[name]) for name in keys):
        raise ValueError('Extraction arrays must contain objects.')
    if key == 'edges':
        for row in data['entities']:
            if not isinstance(row.get('name'), str) or not row['name'].strip():
                raise ValueError('Entity name must be a nonempty string.')
    return data


def string(row, key, default=''):
    value = row.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f'{key} must be a string.')
    return value.strip()
