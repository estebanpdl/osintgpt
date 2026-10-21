'''Serialize a project graph for interchange and graph-database import.'''

import hashlib
import json
import re
import unicodedata

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

from .store import Edge, Entity, GraphStore, merge_key

CYPHERL = '.cypherl'
JSON = '.json'


def _relationship_type(relation: str) -> str:
    normalized = unicodedata.normalize('NFKD', relation or '')
    ascii_parts = ''.join(
        character for character in normalized
        if ord(character) < 128 and not unicodedata.combining(character)
    )
    sanitized = re.sub(r'[^A-Za-z0-9_]+', '_', ascii_parts)
    sanitized = re.sub(r'_+', '_', sanitized).strip('_').upper()
    needs_quoting = '`' in relation or any(
        character.isalnum() and ord(character) >= 128
        for character in normalized
        if not unicodedata.combining(character)
    )
    if needs_quoting and any(character.isalnum() for character in relation):
        escaped = relation.replace('`', '``')

        return f'`{escaped}`'
    if sanitized and not sanitized[0].isdigit():
        return sanitized
    if sanitized:
        return f'RELATED_{sanitized}'

    # Punctuation alone cannot make a useful quoted identifier. A stable
    # suffix keeps even these unusual predicates valid and distinct.
    digest = hashlib.sha256((relation or '').encode('utf-8')).hexdigest()[:12]

    return f'RELATED_{digest.upper()}'


def _cypher_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _entity_line(entity: Entity, project_id='') -> str:
    details = (', n.details = ' + _cypher_string(json.dumps({
        'aliases': entity.aliases, 'description': entity.description,
        'identifiers': entity.identifiers, 'provenance': entity.provenance
    }, ensure_ascii=False, sort_keys=True))) if any((entity.provenance, entity.aliases, entity.description, entity.identifiers)) else ''
    namespace = ', project_id: ' + _cypher_string(project_id) if project_id else ''
    return (
        f'MERGE (n:Entity {{key: {_cypher_string(entity.key)}{namespace}}}) '
        f'SET n.name = {_cypher_string(entity.name)}, '
        f'n.type = {_cypher_string(entity.type)}, '
        f'n.mentions = {int(entity.mentions)}{details}'
    )


def _edge_line(edge: Edge, project_id='') -> str:
    source = _cypher_string(edge.source_id or merge_key(edge.source))
    target = _cypher_string(edge.target_id or merge_key(edge.target))
    relation = _cypher_string(edge.relation)
    ref = _cypher_string(edge.ref)
    evidence = _cypher_string(edge.evidence)
    edge_type = _relationship_type(edge.relation)
    provenance = (', provenance: ' + _cypher_string(json.dumps(edge.provenance, ensure_ascii=False,
                  sort_keys=True))) if edge.provenance else ''
    provenance += ', assertion_id: ' + _cypher_string(edge.assertion_id)
    if (edge.source_id or edge.target_id or edge.relation_polarity != 'unknown' or edge.assertion != 'unknown'
            or edge.evidence_quotes or edge.attributed_to or edge.event_time or edge.statement_time):
        provenance += ''.join(', ' + name + ': ' + _cypher_string(getattr(edge, name))
                              for name in ('relation_polarity', 'assertion', 'attributed_to', 'event_time', 'statement_time'))
        provenance += ', evidence_quotes: ' + _cypher_string(json.dumps(edge.evidence_quotes, ensure_ascii=False))

    # Variables are scoped to one Cypher statement, so every edge line binds
    # its nodes instead of relying on aliases from preceding MERGE lines.
    namespace = ', project_id: ' + _cypher_string(project_id) if project_id else ''
    return (
        f'MATCH (a:Entity {{key: {source}{namespace}}}), '
        f'(b:Entity {{key: {target}{namespace}}}) '
        f'MERGE (a)-[r:{edge_type} '
        f'{{relation: {relation}, ref: {ref}, evidence: {evidence}{provenance}}}]->(b)'
    )


def _snapshot(
    graph: GraphStore,
    refs: Optional[Iterable[str]]
) -> Tuple[List[Entity], List[Edge]]:
    wanted = None if refs is None else tuple(refs)
    edges = graph.edges(refs=wanted) if wanted is not None else graph.edges()
    entities = graph.entities()
    if wanted is None:
        return entities, edges

    endpoint_names = {}
    for edge in edges:
        endpoint_names.setdefault(edge.source_id or merge_key(edge.source), edge.source)
        endpoint_names.setdefault(edge.target_id or merge_key(edge.target), edge.target)

    selected = [entity for entity in entities if entity.key in endpoint_names]
    present = {entity.key for entity in selected}
    selected.extend(
        Entity(key=key, name=name)
        for key, name in endpoint_names.items()
        if key not in present
    )

    return selected, edges


def to_cypherl(
    graph: GraphStore,
    refs: Optional[Iterable[str]] = None,
    *, project=None
) -> str:
    '''Emit independent Cypher statements; supplying project adds verification and export context.'''
    if project is not None:
        from .export_snapshot import graph_package
        return serialize_package(graph_package(graph, project, refs), CYPHERL)
    entities, edges = _snapshot(graph, refs)
    lines = (
        [_entity_line(entity) for entity in entities]
        + [_edge_line(edge) for edge in edges]
    )

    return '\n'.join(f'{line};' for line in lines) + '\n' if lines else ''


def to_json(
    graph: GraphStore,
    refs: Optional[Iterable[str]] = None,
    *, project=None
) -> str:
    '''Emit record-only JSON unless project is supplied for verification and export context.'''
    if project is not None:
        from .export_snapshot import graph_package
        return serialize_package(graph_package(graph, project, refs), JSON)
    entities, edges = _snapshot(graph, refs)

    return json.dumps(
        {
            'entities': [asdict(entity) for entity in entities],
            'edges': [asdict(edge) for edge in edges]
        },
        ensure_ascii=False,
        indent=2
    )


def export_graph(
    graph: GraphStore,
    path: Union[str, Path],
    refs: Optional[Iterable[str]] = None,
    *, project=None
) -> Path:
    '''Accept .json or .cypherl destinations; project enables verification and export context.'''
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == CYPHERL:
        content = to_cypherl(graph, refs, project=project)
    elif suffix == JSON:
        content = to_json(graph, refs, project=project)
    else:
        raise ValueError(
            f'{destination}: graph export must end in {CYPHERL} or {JSON}'
        )

    destination.write_text(content, encoding='utf-8')

    return destination


def serialize_package(package, suffix=JSON):
    if suffix == JSON:
        return json.dumps(package, ensure_ascii=False, sort_keys=True, indent=2)
    if suffix != CYPHERL:
        raise ValueError('Graph export must end in .cypherl or .json')
    metadata = json.dumps(package['metadata'], ensure_ascii=False, sort_keys=True)
    identity = hashlib.sha256(json.dumps(package, ensure_ascii=False, sort_keys=True).encode('utf8')).hexdigest()
    project_id = package['metadata']['project']['id']
    lines = [f'MERGE (export:OsintGPTExport {{id: {_cypher_string(identity)}}}) SET export.metadata = {_cypher_string(metadata)}']
    lines += [_entity_line(Entity(**row), project_id) for row in package['entities']]
    for row in package['edges']:
        edge = Edge(**{key: value for key, value in row.items() if key in Edge.__dataclass_fields__})
        object.__setattr__(edge, 'assertion_id', row['assertion_id'])
        fields = {'verification': row['verification'], 'source_version': row['source_version'],
                  'citation': row['citation'], 'source_ref': row['source_ref'], 'locations': row['locations'],
                  'traversal': row.get('traversal'), 'export_id': identity}
        assignments = ', '.join(f'r.{key} = ' + _cypher_string(
            json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) or value is None else value)
            for key, value in fields.items())
        lines.append(_edge_line(edge, project_id) + ' SET ' + assignments)
    return '\n'.join(line + ';' for line in lines) + '\n'
