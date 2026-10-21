'''Conservative entity resolution with persistent IDs and bounded source context.'''

import hashlib

from .grounding import GroundingError, string
from .records import Entity, entity_origin, merge_entity, merge_key
from .windowing import encode, units
from .entity_resolution import decide, shared_identifier
from osintgpt.ingestion.quotation_projection import supported_wording


class EntityRegistry:
    def __init__(self, entities=()):
        self.entities = {entity.key: entity for entity in entities}
        self.candidate_evidence = {}
        self.warnings = []

    def context(self, document, snapshot, window, budget):
        text = merge_key(document.text[window.start:window.end])
        ranked = []
        for entity in self.entities.values():
            origin = entity_origin(entity, document.ref, snapshot.version)
            same_document = origin is not None
            mentioned = any(merge_key(name) in text for name in (entity.name, *entity.aliases) if name)
            if mentioned or same_document:
                ranked.append((mentioned, same_document, entity))
        ranked.sort(key=lambda item: (item[0], item[1],
                    max((m['end'] for m in item[2].provenance.get('matches', [])), default=0)), reverse=True)
        entries, ranges = [], []
        for _, same_document, entity in ranked:
            entry = {'id': entity.key, 'name': entity.name, 'type': entity.type,
                     'aliases': list(entity.aliases[-12:]), 'description': entity.description,
                     'identifiers': [{k: item[k] for k in ('issuer', 'value')} for item in entity.identifiers[-8:]],
                     'document_ref': entity.provenance.get('document_ref', ''),
                     'source_version': entity.provenance.get('source_version', '')}
            reference = None
            origin = entity_origin(entity, document.ref, snapshot.version)
            if origin and origin.get('matches'):
                entry.update(document_ref=document.ref, source_version=snapshot.version)
                match = origin['matches'][-1]
                start, end = max(0, match['start'] - 120), min(len(document.text), match['end'] + 120)
                if end - start <= 1000:
                    entry['source_context'] = {'start': start, 'end': end, 'text': document.text[start:end]}
                    reference = (start, end)
            if units(encode(entries + [entry])) > budget:
                continue
            entries.append(entry)
            if reference:
                ranges.append(reference)
            if len(entries) == 32:
                break
        return entries, ranges

    def resolve(self, rows, scope, allowed):
        resolved, by_local = {}, {}
        for index, row in enumerate(rows):
            name = string(row, 'name')
            local = string(row, 'id', name)
            if not name or not local or local in by_local:
                raise ValueError('Entities require distinct local IDs and nonempty names.')
            entity_type = string(row, 'type')
            quotes, provenance, warnings = scope.supported(row.get('evidence', name))
            if not supported_wording(name, quotes, scope.document):
                raise GroundingError('Entity name must occur in its supporting quotation.')
            aliases, identifiers = [], []
            for field, validator, kept in (('aliases', self._aliases, aliases), ('identifiers', self._identifiers, identifiers)):
                values = row.get(field, [])
                if not isinstance(values, list):
                    warnings.append(f'{field} must be a list; optional field discarded.')
                    continue
                for value in values:
                    try:
                        kept.extend(validator([value], scope))
                    except (ValueError, TypeError) as error:
                        warnings.append(f'{field}: {error}')
            existing, decision = decide(self, row, scope, allowed, identifiers, entity_type, name)
            if decision['action'] == 'unresolved':
                warnings.append('Identity unresolved: ' + decision['reason'])
            # Stable across model-local ID changes and source-version metadata.
            identity = encode([scope.document.ref, name, entity_type, quotes, provenance['matches'][0]['start']])
            key = existing.key if existing else 'ent_' + hashlib.sha256(identity.encode()).hexdigest()[:32]
            mention = {**provenance, 'name': name, 'type': entity_type, 'local_id': local,
                       'window': getattr(self, 'window', 0), 'decision': decision}
            mention['mention_id'] = 'mention_' + hashlib.sha256(encode(mention).encode()).hexdigest()
            entity = Entity(key, name, entity_type, 1, tuple(alias['name'] for alias in aliases),
                            string(row, 'description'), tuple(identifiers),
                            {**provenance, 'alias_evidence': aliases, 'source_name': name,
                             'mentions': [mention], 'resolution': decision, 'field_warnings': warnings})
            if existing:
                entity = merge_entity(existing, entity)
            by_local[local] = key
            resolved[key] = entity
            self.warnings.extend((row, warning) for warning in warnings)
        return resolved, by_local

    @staticmethod
    def _shared_identifier(entity, identifiers, entity_type):
        return shared_identifier(entity, identifiers, entity_type)

    @staticmethod
    def _aliases(rows, scope):
        if not isinstance(rows, list):
            raise ValueError('Aliases must be a list of supported names.')
        aliases = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Each alias needs a name and evidence.')
            name = string(row, 'name')
            quotes, provenance = scope.prove(row.get('evidence'))
            if not name or not supported_wording(name, quotes, scope.document):
                raise GroundingError('Alias name not present in its evidence.')
            aliases.append({'name': name, 'provenance': provenance})
        return aliases

    @staticmethod
    def _identifiers(rows, scope):
        if not isinstance(rows, list):
            raise ValueError('Identifiers must be a list of supported values.')
        result = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Identifiers require issuer, value, and evidence.')
            issuer, value = string(row, 'issuer'), string(row, 'value')
            quotes, provenance = scope.prove(row.get('evidence'))
            if not issuer or not value or not all(supported_wording(item, quotes, scope.document) for item in (issuer, value)):
                raise GroundingError('Identifier issuer and value must occur in its evidence.')
            result.append({'issuer': issuer, 'value': value, 'provenance': provenance})
        return result

    def commit(self, entities):
        self.entities.update({entity.key: entity for entity in entities})
