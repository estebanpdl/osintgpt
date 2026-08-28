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
    if sanitized and not sanitized[0].isdigit():
        return sanitized
    if sanitized:
        return f'RELATED_{sanitized}'

    # A shared RELATED fallback would collapse the type distinction for every
    # predicate written outside ASCII. The original remains a property, while
    # this stable suffix keeps its database type distinct and reproducible.
    digest = hashlib.sha256((relation or '').encode('utf-8')).hexdigest()[:12]

    return f'RELATED_{digest.upper()}'


def _cypher_string(value: object) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _entity_line(entity: Entity) -> str:
    return (
        f'MERGE (n:Entity {{key: {_cypher_string(entity.key)}}}) '
        f'SET n.name = {_cypher_string(entity.name)}, '
        f'n.type = {_cypher_string(entity.type)}, '
        f'n.mentions = {int(entity.mentions)}'
    )


def _edge_line(edge: Edge) -> str:
    source = _cypher_string(merge_key(edge.source))
    target = _cypher_string(merge_key(edge.target))
    relation = _cypher_string(edge.relation)
    ref = _cypher_string(edge.ref)
    evidence = _cypher_string(edge.evidence)
    edge_type = _relationship_type(edge.relation)

    # Variables are scoped to one Cypher statement, so every edge line binds
    # its nodes instead of relying on aliases from preceding MERGE lines.
    return (
        f'MATCH (a:Entity {{key: {source}}}), '
        f'(b:Entity {{key: {target}}}) '
        f'MERGE (a)-[r:{edge_type} '
        f'{{relation: {relation}, ref: {ref}, evidence: {evidence}}}]->(b)'
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
        endpoint_names.setdefault(merge_key(edge.source), edge.source)
        endpoint_names.setdefault(merge_key(edge.target), edge.target)

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
    refs: Optional[Iterable[str]] = None
) -> str:
    '''
    Serialize a graph as independent Cypher statements, one per line.

    Args:
        graph (GraphStore): Graph to serialize.
        refs (Iterable[str], optional): Restrict edges and their entities to \
            these documents.

    Returns:
        str: CYPHERL ready for mgconsole or cypher-shell.
    '''
    entities, edges = _snapshot(graph, refs)
    lines = (
        [_entity_line(entity) for entity in entities]
        + [_edge_line(edge) for edge in edges]
    )

    return '\n'.join(lines) + '\n' if lines else ''


def to_json(
    graph: GraphStore,
    refs: Optional[Iterable[str]] = None
) -> str:
    '''
    Serialize graph records as readable UTF-8 JSON.

    Args:
        graph (GraphStore): Graph to serialize.
        refs (Iterable[str], optional): Restrict edges and their entities to \
            these documents.

    Returns:
        str: JSON containing entity and edge arrays.
    '''
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
    refs: Optional[Iterable[str]] = None
) -> Path:
    '''
    Write a graph in the format selected by the destination suffix.

    Args:
        graph (GraphStore): Graph to export.
        path (Union[str, Path]): Destination ending in .cypherl or .json.
        refs (Iterable[str], optional): Restrict the exported documents.

    Raises:
        ValueError: If the destination suffix is unsupported.

    Returns:
        Path: The written destination.
    '''
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == CYPHERL:
        content = to_cypherl(graph, refs)
    elif suffix == JSON:
        content = to_json(graph, refs)
    else:
        raise ValueError(
            f'{destination}: graph export must end in {CYPHERL} or {JSON}'
        )

    destination.write_text(content, encoding='utf-8')

    return destination
