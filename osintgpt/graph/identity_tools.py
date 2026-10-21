'''Bounded extraction-time entity lookup; no provider construction or graph writes.'''

from copy import deepcopy

from .calling import generation_provider, response_hooks
from .contract import EXTRACTION_SCHEMA, _object, _array, _TEXT, _QUOTES
from .entity_catalog import EntityCatalog, evidence_entries
from .windowing import encode, units

RESOLVING_SCHEMA = deepcopy(EXTRACTION_SCHEMA)
RESOLUTION_SCHEMA = _object({
    'action': {'type': 'string', 'enum': ['create', 'reuse', 'unresolved']},
    'method': {'type': 'string', 'enum': ['identifier', 'context', 'explicit_alias', 'same_source', 'new', 'unresolved']},
    'reason': _TEXT, 'evidence': _QUOTES, 'candidate_evidence': _QUOTES,
    'type_compatible': {'type': 'boolean'}, 'type_reason': _TEXT,
})
RESOLVING_SCHEMA['properties']['entities']['items']['properties']['resolution'] = RESOLUTION_SCHEMA
RESOLVING_SCHEMA['properties']['entities']['items']['required'].append('resolution')

TOOLS = [
    {'name': 'find_entity_candidates', 'description':
     'Search existing and staged identities by names, aliases or identifiers. Refine or page truncated results. '
     'A name match is a candidate, not proof of identity. Empty entity_type means any type.',
     'parameters': _object({'queries': _array(_TEXT), 'entity_type': _TEXT,
                            'offset': {'type': 'integer'}, 'limit': {'type': 'integer'}})},
    {'name': 'read_entity_context', 'description':
     'Read saved sourced quotations for a candidate ID before proposing reuse. Page for more evidence. '
     'Use next_offset and next_quote_offset to continue long quotations; initially both offsets are zero. '
     'These quotations support identity only, not new relationships in the current document.',
     'parameters': _object({'entity_id': _TEXT, 'offset': {'type': 'integer'}, 'limit': {'type': 'integer'},
                            'quote_offset': {'type': 'integer'}})},
]
SUBMIT = {'name': 'submit_extraction', 'description':
          'Finish this source window with the complete structured extraction. Call once, alone, after any identity lookups. '
          'This proposes data for validation; it does not write to the graph.', 'parameters': RESOLVING_SCHEMA}


def interactive_supported(generator):
    provider = generation_provider(generator)
    return (getattr(provider, 'supports_tools', False) and getattr(provider, 'supports_structured_output', False)
            and {'tools', 'history'} <= response_hooks(generator).keys())


class IdentityTools:
    def __init__(self, registry, allowance=None):
        self.catalog = EntityCatalog(registry.entities.values())
        self.allowance = allowance
        self.allowed, self.evidence = set(), {}

    def run(self, name, arguments):
        try:
            return self._run(name, arguments)
        except (ValueError, TypeError, KeyError) as error:
            return {'status': 'invalid_arguments', 'problem': str(error), 'complete': False}

    def _run(self, name, args):
        if not isinstance(args, dict):
            raise ValueError('Tool arguments must be an object.')
        offset, limit = args.get('offset', 0), args.get('limit', 10)
        if type(offset) is not int or offset < 0 or type(limit) is not int or limit < 1:
            raise ValueError('offset must be nonnegative and limit must be positive.')
        if name == 'find_entity_candidates':
            queries = args.get('queries')
            if (not isinstance(queries, list) or not queries
                    or any(not isinstance(q, str) or not q.strip() for q in queries)):
                raise ValueError('Supply a nonempty list of nonempty queries.')
            entity_type = args.get('entity_type', '')
            if not isinstance(entity_type, str):
                raise ValueError('entity_type must be text.')
            results = []
            for query in queries:
                entities, mode, total = self.catalog.search(query, offset=offset, limit=limit, entity_type=entity_type)
                result = {'query': query, 'match': mode, 'total': total, 'offset': offset, 'candidates': []}
                for entity in entities:
                    row = {'id': entity.key, 'name': entity.name, 'type': entity.type,
                           'aliases': list(entity.aliases), 'description': entity.description,
                           'details_truncated': False,
                           'evidence_tool': 'read_entity_context'}
                    if self.allowance is not None and units(encode([*results, {**result, 'candidates': result['candidates'] + [row]}])) > self.allowance:
                        break
                    result['candidates'].append(row)
                    self.allowed.add(entity.key)
                result['next_offset'] = offset + len(result['candidates'])
                result['complete'] = result['next_offset'] >= total
                if not result['candidates'] and total > offset:
                    result['note'] = 'Candidates did not fit; refine the search. This is not a no-match result.'
                results.append(result)
            return {'status': 'ok', 'catalogue_version': self.catalog.version, 'results': results}
        if name == 'read_entity_context':
            key = args.get('entity_id')
            if key not in self.allowed:
                raise ValueError('Search for this candidate before reading its evidence.')
            entity = self.catalog.entities[key]
            all_entries, entries = evidence_entries(entity), []
            quote_offset = args.get('quote_offset', 0)
            if type(quote_offset) is not int or quote_offset < 0:
                raise ValueError('quote_offset must be a nonnegative character offset.')
            end, next_quote = offset, quote_offset
            for item in all_entries[offset:offset + limit]:
                quote = item['quote']
                if quote_offset >= len(quote):
                    raise ValueError('quote_offset is outside the selected quotation.')
                def portion(length):
                    excerpt = quote[quote_offset:quote_offset + length]
                    start = item['location']['start'] + quote_offset
                    return {**item, 'quote': excerpt, 'partial': quote_offset > 0 or length < len(quote),
                            'quote_offset': quote_offset, 'location': {**item['location'],
                            'start': start, 'end': start + length, 'source_excerpt': excerpt}}
                low, high = 0, len(quote) - quote_offset
                while low < high:
                    middle = (low + high + 1) // 2
                    if self.allowance is None or units(encode(entries + [portion(middle)])) <= self.allowance:
                        low = middle
                    else:
                        high = middle - 1
                if not low:
                    break
                entries.append(portion(low))
                if quote_offset + low < len(quote):
                    next_quote = quote_offset + low
                    break
                end += 1
                quote_offset = next_quote = 0
            self.evidence.setdefault(key, set()).update(item['quote'] for item in entries)
            return {'status': 'ok', 'entity_id': key, 'evidence': entries, 'total': len(all_entries),
                    'next_offset': end, 'next_quote_offset': next_quote, 'complete': end >= len(all_entries),
                    'note': 'No evidence fits; refine or leave unresolved.' if not entries and end < len(all_entries) else ''}
        raise ValueError('Unknown extraction tool; only entity search and evidence reads are available.')


def extract_with_lookup(calls, system, payload, registry):
    from .identity_session import extract_with_lookup as run
    return run(calls, system, payload, registry)
