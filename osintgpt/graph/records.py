'''Graph identities, qualified claims, and extraction audit records.'''

import unicodedata
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone


def merge_key(name):
    cleaned = unicodedata.normalize('NFC', (name or '').strip())
    return ' '.join(cleaned.strip('.,;:!?"\'()[]{}«»„“”‘’').casefold().split())


def normalize_relation_polarity(data):
    '''Read old application records without rewriting stored data or original model responses.'''
    if 'polarity' in data and 'relation_polarity' in data and data['polarity'] != data['relation_polarity']:
        raise ValueError('Conflicting polarity and relation_polarity values in saved graph data.')
    # Keep position as well as value: repair request hashes include serialized candidates.
    return {('relation_polarity' if key == 'polarity' else key): value for key, value in data.items()}


@dataclass(frozen=True)
class Entity:
    key: str
    name: str
    type: str = ''
    mentions: int = 0
    aliases: tuple = ()
    description: str = ''
    identifiers: tuple = ()
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    ref: str
    evidence: str
    provenance: dict = field(default_factory=dict)
    source_id: str = ''
    target_id: str = ''
    relation_polarity: str = 'unknown'
    assertion: str = 'unknown'
    attributed_to: str = ''
    event_time: str = ''
    statement_time: str = ''
    evidence_quotes: tuple = ()
    assertion_id: str = ''

    @classmethod
    def from_dict(cls, data):
        data = normalize_relation_polarity(data)
        return cls(**{**data, 'evidence_quotes': tuple(data.get('evidence_quotes', []))})

    def __post_init__(self):
        identity = [self.source_id or merge_key(self.source), self.relation,
                    self.target_id or merge_key(self.target), self.ref,
                    self.provenance.get('source_version', ''), self.evidence_quotes or (self.evidence,),
                    self.relation_polarity, self.assertion, self.attributed_to, self.event_time, self.statement_time,
                    self.provenance.get('supporting', self.provenance.get('matches', []))]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                           separators=(',', ':')).encode()).hexdigest()
        object.__setattr__(self, 'assertion_id', 'assert_' + digest)

    @property
    def sentence(self):
        qualification = ', '.join(item for item in (self.relation_polarity, self.assertion) if item != 'unknown')
        details = [qualification] if qualification else []
        details.extend(f'{label}: {value}' for label, value in (
            ('attributed to', self.attributed_to), ('event', self.event_time),
            ('statement', self.statement_time)) if value)
        return f'{self.source} — {self.relation} → {self.target}' + (f' [{"; ".join(details)}]' if details else '')


@dataclass(frozen=True)
class RejectedClaim:
    ref: str
    source_ref: str
    source_version: str
    window: int
    candidate: dict
    reason: str
    status: str = 'rejected'
    repair_problem: str = ''
    corrected_evidence: tuple = ()
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    supplied_ranges: tuple = ()


def entity_origin(entity, document_ref, version):
    origins = [entity.provenance, *entity.provenance.get('identity_sources', [])]
    return next((item for item in origins if item.get('document_ref') == document_ref
                 and item.get('source_version') == version), None)


def merge_entity(previous, current):
    '''Keep identity evidence when a later window or document reuses an ID.'''
    origins = [*previous.provenance.get('identity_sources', []), previous.provenance,
               *current.provenance.get('identity_sources', []), current.provenance]
    sources = {}
    for origin in origins:
        if origin.get('source_version'):
            key = (origin.get('document_ref'), origin['source_version'])
            sources[key] = {name: origin[name] for name in
                            ('source_ref', 'document_ref', 'source_version', 'matches') if name in origin}
    aliases = [*previous.provenance.get('alias_evidence', []), *current.provenance.get('alias_evidence', [])]
    mentions = {item['mention_id']: item for item in
                [*previous.provenance.get('mentions', []), *current.provenance.get('mentions', [])]}
    return replace(current, name=previous.name, type=previous.type or current.type,
                   aliases=tuple(dict.fromkeys((*previous.aliases, *current.aliases,
                                               *([current.name] if current.name != previous.name else [])))),
                   identifiers=tuple({(item['issuer'], item['value']): item
                                      for item in (*previous.identifiers, *current.identifiers)}.values()),
                   provenance={**current.provenance, 'identity_sources': list(sources.values()),
                               'alias_evidence': list({str(item): item for item in aliases}.values()),
                               'mentions': list(mentions.values())})
