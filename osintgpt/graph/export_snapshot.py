'''Provider-free export snapshots retain query evidence and distinguish full from selected graphs.'''

from copy import deepcopy
from dataclasses import asdict

from .retrieval import claim, retrieval_status
from .state import SCHEMA_VERSION, refresh_status
from .store import Entity, GraphStore, merge_key
from .verify import verify_claims
from .model_metadata import published_models, selected_models
from .identity_storage import identity_links, resolutions


def project_identity(project):
    return {'id': project.id, 'slug': project.slug, 'name': project.name}


def _entities(edges, entities=()):
    wanted = {row[side + '_id'] or merge_key(row[side]): row[side]
              for row in edges for side in ('source', 'target')}
    known = {row['key']: row for row in entities}
    return [deepcopy(known.get(key, asdict(Entity(key, wanted[key])))) for key in sorted(wanted)]


def _versions(edges):
    return sorted({(row.get('source_ref', row['ref']), row.get('source_version', '')) for row in edges})


def graph_package(graph, project, refs=None):
    refs = list(refs) if refs is not None else None
    owns_transaction = not graph.connection.in_transaction
    if owns_transaction:
        graph.connection.execute('BEGIN')
    try:
        edges = graph.edges(refs=refs)
        entities = [asdict(entity) for entity in graph.entities()]
        saved = graph.state.read()
        versions = graph.state.versions()
        models = published_models(graph.connection, refs=refs)
        identity_context = {'links': identity_links(graph.connection), 'decisions': resolutions(graph.connection)}
        refresh_status(project, graph.state, saved)
    finally:
        if owns_transaction:
            graph.connection.rollback()
    state = retrieval_status(project, allow_disabled=True, saved=saved)
    checked = verify_claims(project, edges, first_only=True, require_registered=True)
    rows = [claim(edge, check) for edge, check in zip(edges, checked.results)]
    rows.sort(key=lambda row: row['assertion_id'])
    full_entities = {row['key']: row for row in entities}
    full_entities.update({row['key']: row for row in _entities(rows, entities)})
    metadata = {'format_version': 1, 'graph_schema_version': SCHEMA_VERSION,
                'project': project_identity(project), 'scope': 'sources' if refs is not None else 'full',
                'graph': state, 'filters': {'sources': list(refs)} if refs is not None else {},
                'verification_basis': 'export_preparation', 'truncated': False,
                'source_versions': [{'ref': ref, 'source_version': version} for ref, version in
                                    sorted(set(_versions(rows)) | {(ref, version) for ref, version in versions.items()
                                                                  if refs is None or ref in refs})],
                'build_counts': saved['counts'], 'build_problem': saved['problem'],
                'extraction_models': models, 'run_model': saved['usage'].get('graph_model', {}),
                'identity_resolution': identity_context}
    return {'metadata': metadata,
            'entities': _entities(rows, entities) if refs is not None else [full_entities[key] for key in sorted(full_entities)],
            'edges': rows}


def full_graph_package(project):
    path = project.paths.root / 'graph.sqlite'
    if not path.is_file():
        raise ValueError(f'{project.slug}: graph has not been built')
    with GraphStore(path, read_only=True) as graph:
        if not graph.is_built:
            raise ValueError(f'{project.slug}: graph has not been built')
        return graph_package(graph, project)


def query_export_context(project, result, *, request=None):
    rows = result.get('path', result.get('claims', []))
    identities = _entities(rows)
    models = {}
    if identities:
        with GraphStore(project.paths.root / 'graph.sqlite', read_only=True) as graph:
            graph.connection.execute('BEGIN')
            models = published_models(graph.connection, versions=_versions(rows))
            known = {}
            for entity in graph.entities_by_ids(row['key'] for row in identities):
                value = asdict(entity)
                for key in (entity.key, *entity.provenance.get('member_ids', [])):
                    known[key] = value
            for index, row in enumerate(identities):
                if row['key'] in known:
                    identities[index] = deepcopy(known[row['key']])
    return {'project': project_identity(project), 'entities': identities, 'request': deepcopy(request or {}),
            'extraction_models': models}


def selection_package(project, result, *, context, claims=None, filters=None):
    if context['project']['id'] != project.id:
        raise ValueError('The selection belongs to a different project.')
    original = result.get('path', result.get('claims', []))
    selected = original if claims is None else claims
    original_by_id = {row['assertion_id']: row for row in original}
    if any(row != original_by_id.get(row['assertion_id']) for row in selected):
        raise ValueError('Export selection must contain unchanged query assertions.')
    path = 'path' in result
    filtered = [row['assertion_id'] for row in selected] != [row['assertion_id'] for row in original]
    query = {key: value for key, value in result.items() if key not in ('path', 'claims')}
    query['request'] = context.get('request', {})
    metadata = {'format_version': 1, 'graph_schema_version': SCHEMA_VERSION,
                'project': context['project'], 'scope': 'filtered' if filtered or filters else 'path' if path else 'neighborhood',
                'graph': result.get('graph', {}), 'query': query, 'filters': filters or {},
                'verification_basis': 'displayed_query', 'truncated': result.get('truncated', False),
                'path_complete': bool(path and not filtered and result.get('path_complete', False)),
                'source_versions': [{'ref': ref, 'source_version': version} for ref, version in _versions(selected)],
                'extraction_models': selected_models(context.get('extraction_models', {}), _versions(selected))}
    return deepcopy({'metadata': metadata, 'entities': _entities(selected, context['entities']), 'edges': selected})
