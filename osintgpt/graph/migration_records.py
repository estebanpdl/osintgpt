'''Recognize supported graph layouts and convert their stored records without extraction.'''

import json
from dataclasses import asdict

from osintgpt.ingestion import Corpus
from osintgpt.ingestion.source_view import reference_for, reference_parts

from .records import Edge, Entity, normalize_relation_polarity
from .store import GraphStore
from .state import GraphState, SCHEMA_VERSION

LEGACY_NOTE = ('Migrated legacy records lack the newer identity and evidence guarantees. '
               'Missing qualifications remain unknown; rebuild registered sources to re-extract them. '
               'Migration makes no model calls and cannot reconstruct historical source text or usage.')


RENAME_NOTE = (
    'Schema 8 renames polarity to relation_polarity in published assertions and saved accepted windows. '
    'Values, IDs, evidence, identity decisions, recipes, counts, and usage are preserved. '
    'Original responses and historical diagnostics retain their original spelling. '
    'The rename alone requires no provider calls or re-extraction and does not correct model interpretation.'
)


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _layout(connection):
    return {row[0]: [tuple(column[1:]) for column in connection.execute(
        f'PRAGMA table_info({_quote(row[0])})')]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _unique_indexes(connection, tables):
    return {table: {tuple(col[2] for col in connection.execute(f'PRAGMA index_info({_quote(index[1])})'))
                    for index in connection.execute(f'PRAGMA index_list({_quote(table)})') if index[2]}
            for table in tables}


def inspect_schema(connection):
    version = connection.execute('PRAGMA user_version').fetchone()[0]
    if connection.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
        raise ValueError('Graph integrity check failed; no migration was applied.')
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type IN ('trigger', 'view')").fetchone():
        raise ValueError('Unrecognized graph layout: views or triggers are present; no migration was applied.')
    layout = _layout(connection)
    with GraphStore(':memory:') as template:
        current = _layout(template.connection)
        unique = _unique_indexes(template.connection, current)
    if version == 0:
        expected = {
            'entities': [('key', 'TEXT', 0, None, 1), ('name', 'TEXT', 1, None, 0),
                         ('type', 'TEXT', 1, "''", 0), ('mentions', 'INTEGER', 1, '0', 0)],
            'edges': [('id', 'INTEGER', 0, None, 1), ('source_key', 'TEXT', 1, None, 0),
                      ('target_key', 'TEXT', 1, None, 0), ('relation', 'TEXT', 1, None, 0),
                      ('ref', 'TEXT', 1, None, 0), ('evidence', 'TEXT', 1, None, 0)],
        }
        if not layout:
            return version
        unique = {'entities': {('key',)}, 'edges': {('source_key', 'target_key', 'relation', 'ref', 'evidence')}}
    elif version in (5, 6):
        expected = {name: columns for name, columns in current.items()
                    if name not in {'entity_links', 'identity_decisions'}}
        unique = {name: values for name, values in unique.items() if name in expected}
        if version == 5:
            expected['edges'] = [col for col in current['edges'] if col[0] != 'assertion_id']
            unique['edges'].remove(('assertion_id',))
    elif version in (7, SCHEMA_VERSION):
        expected = current
    else:
        raise ValueError(f'Unsupported graph schema version {version}. Supported migrations: 0, 5, 6 and 7 to {SCHEMA_VERSION}. '
                         'The graph was not changed.')
    if (layout != expected or _unique_indexes(connection, layout) != unique
            or any(connection.execute(f'PRAGMA foreign_key_list({_quote(table)})').fetchone() for table in layout)):
        raise ValueError(f'Unrecognized layout for graph schema version {version}; the graph was not changed.')
    return version


def counts(connection):
    tables = _layout(connection)
    return {name: connection.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] if name in tables else 0
            for name in ('entities', 'edges')}


def _create_tables(connection, wanted):
    with GraphStore(':memory:') as template:
        statements = template.connection.execute(
            "SELECT tbl_name, sql FROM sqlite_master WHERE sql IS NOT NULL "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END").fetchall()
    for table, sql in statements:
        if table in wanted:
            connection.execute(sql)
    if 'graph_build' in wanted:
        connection.execute('INSERT INTO graph_build (id) VALUES (1)')


def _rename(connection, table):
    indexes = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)).fetchall()
    for (name,) in indexes:
        connection.execute(f'DROP INDEX {_quote(name)}')
    connection.execute(f'ALTER TABLE {table} RENAME TO migration_{table}')


def _source_refs(project, refs):
    corpus = Corpus.load(project.paths.sources)
    covered = {path.resolve() for path in corpus.files(project.paths.root)}
    covered.update((project.paths.root / source.path).resolve() for source in corpus)
    resolved = {}
    for ref in refs:
        path, _ = reference_parts(ref, project.paths.root, covered)
        resolved[ref] = reference_for(path, project.paths.root) if path else ref
    return resolved


def _original_records(connection, project):
    for row in connection.execute('SELECT * FROM migration_entities'):
        entity = Entity(**dict(row), provenance={'legacy_schema': 0, 'requires_reextraction': True})
        connection.execute('INSERT INTO entities VALUES (?, ?, ?, ?, ?)',
                           (entity.key, entity.name, entity.type, entity.mentions, json.dumps(asdict(entity))))
    refs = _source_refs(project, [row[0] for row in connection.execute('SELECT DISTINCT ref FROM migration_edges')])
    for row in connection.execute('''SELECT e.*, s.name AS source, t.name AS target FROM migration_edges e
            JOIN entities s ON s.key=e.source_key JOIN entities t ON t.key=e.target_key ORDER BY e.id'''):
        edge = Edge(source=row['source'], target=row['target'], source_id=row['source_key'],
                    target_id=row['target_key'], relation=row['relation'], ref=row['ref'], evidence=row['evidence'],
                    provenance={'source_ref': refs[row['ref']], 'legacy_schema': 0, 'requires_reextraction': True})
        _insert_edge(connection, row, edge, refs[row['ref']])
    for ref in set(refs.values()):
        connection.execute("INSERT INTO graph_documents (ref, fingerprint, quality_gaps) VALUES (?, '', 1)", (ref,))
    if any(counts(connection).values()):
        connection.execute("UPDATE graph_build SET status='complete_with_gaps', available=1, problem=?, usage=?",
                           (LEGACY_NOTE, json.dumps({'complete': False, 'note': 'Historical usage was not recorded.'})))


def _insert_edge(connection, row, edge, source_ref):
    connection.execute('''INSERT INTO edges
        (id, source_key, target_key, relation, ref, evidence, source_ref, data, assertion_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (row['id'], row['source_key'], row['target_key'], row['relation'], row['ref'], row['evidence'],
         source_ref, json.dumps(asdict(edge), ensure_ascii=False, sort_keys=True), edge.assertion_id))


def rename_fields(connection, *, apply=False):
    '''Use the same conversion for dry-run validation and transactional application.

    Only known accepted-edge locations are application-owned canonical records.
    Rejected candidates, response/tool journals, provenance, and identity/build
    histories are audit material: do not recursively rename arbitrary JSON keys.
    '''
    counts = {'published_assertions': 0, 'saved_assertions': 0, 'saved_windows': 0}
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ('edges', 'graph_windows'):
        if table not in tables:
            continue
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info({table})')}
        if 'data' not in columns:  # Original unversioned layout is handled by its existing converter.
            continue
        for key, raw in connection.execute(f'SELECT rowid, data FROM {table}'):
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError('Expected a structured object.')
                if table == 'edges':
                    updated = normalize_relation_polarity(data)
                    counts['published_assertions'] += int(updated != data)
                else:
                    edges = data['edges']
                    if not isinstance(edges, list) or any(not isinstance(edge, dict) for edge in edges):
                        raise ValueError('Saved accepted assertions must be a list of objects.')
                    normalized = [normalize_relation_polarity(edge) for edge in edges]
                    counts['saved_assertions'] += sum(old != new for old, new in zip(edges, normalized))
                    updated = {**data, 'edges': normalized}
                    counts['saved_windows'] += int(updated != data)
                if apply and updated != data:
                    connection.execute(f'UPDATE {table} SET data=? WHERE rowid=?',
                                       (json.dumps(updated, ensure_ascii=False, sort_keys=True), key))
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f'{table} row {key}: {error}') from error
    return counts


def upgrade_records(connection, project, version):
    before = counts(connection)
    tables = _layout(connection)
    if version == 0:
        for table in tables:
            _rename(connection, table)
        with GraphStore(':memory:') as template:
            _create_tables(connection, _layout(template.connection))
        if tables:
            _original_records(connection, project)
            connection.execute('DROP TABLE migration_edges')
            connection.execute('DROP TABLE migration_entities')
    elif version == 5:
        _rename(connection, 'edges')
        _create_tables(connection, {'edges'})
        for row in connection.execute('SELECT * FROM migration_edges ORDER BY id'):
            edge = Edge.from_dict(json.loads(row['data']))
            _insert_edge(connection, row, edge, row['source_ref'])
        connection.execute('DROP TABLE migration_edges')
    if version in (5, 6):
        _create_tables(connection, {'entity_links', 'identity_decisions'})
    rename_fields(connection, apply=True)
    connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
    validate_records(connection, before)


def validate_records(connection, before):
    if counts(connection) != before:
        raise ValueError('Migration changed entity or relationship counts; rolling back.')
    if connection.execute('''SELECT 1 FROM edges e LEFT JOIN entities s ON s.key=e.source_key
            LEFT JOIN entities t ON t.key=e.target_key WHERE s.key IS NULL OR t.key IS NULL LIMIT 1''').fetchone():
        raise ValueError('Graph contains relationships with missing entities; rolling back.')
    for (data,) in connection.execute('SELECT data FROM entities'):
        Entity(**json.loads(data))
    for (data,) in connection.execute('SELECT data FROM edges'):
        Edge.from_dict(json.loads(data))
    GraphState(connection, ':memory:').read()
    previous_factory = connection.row_factory
    try:
        connection.row_factory = None
        inspect_schema(connection)
    finally:
        connection.row_factory = previous_factory
