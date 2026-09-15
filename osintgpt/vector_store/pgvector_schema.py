'''Postgres schema creation and additive index provenance migration.'''

# HNSW uses cosine distance, matching the store's retrieval contract.
INDEX_SUFFIX = '_vector_idx'


def prepare(store) -> None:
    '''
    The extension has to exist before a vector column can. Creating it
    needs privileges an operator may not have granted, so the failure says
    what to run rather than what went wrong.
    '''
    try:
        with store.connection.cursor() as cursor:
            cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        store.connection.commit()
    except Exception as error:
        store.connection.rollback()
        raise RuntimeError(
            'the pgvector extension is not available: ask a database '
            'owner to run CREATE EXTENSION vector on this database'
        ) from error

    store.register_vector(store.connection)
    if store._table_exists():
        with store.connection.cursor() as cursor:
            cursor.execute(store._sql(
                "ALTER TABLE {} ADD COLUMN IF NOT EXISTS document_ref text NOT NULL DEFAULT ''"
            ))
        store.connection.commit()


def ensure_table(store, cursor, dimensions: int) -> None:
    '''
    Create on first write, when the vector size is finally known.
    '''
    cursor.execute(store._sql(
        'CREATE TABLE IF NOT EXISTS {} ('
        '  id              bigserial PRIMARY KEY,'
        '  ref             text NOT NULL,'
        '  sequence        integer NOT NULL,'
        '  text            text NOT NULL,'
        '  path            text NOT NULL DEFAULT \'\','
        '  "timestamp"     text NOT NULL DEFAULT \'\','
        '  author          text NOT NULL DEFAULT \'\','
        '  metadata        jsonb NOT NULL DEFAULT \'{{}}\'::jsonb,'
        '  embedding_model text NOT NULL,'
        '  document_ref    text NOT NULL DEFAULT \'\','
        f'  vector          vector({dimensions}) NOT NULL'
        ')'
    ))
    cursor.execute(store._sql(
        'CREATE INDEX IF NOT EXISTS ' + store.table + '_ref_idx '
        'ON {} (ref)'
    ))
    cursor.execute(store._sql(
        'CREATE INDEX IF NOT EXISTS ' + store.table + '_model_idx '
        'ON {} (embedding_model)'
    ))
    cursor.execute(store._sql(
        'CREATE INDEX IF NOT EXISTS ' + store.table + INDEX_SUFFIX + ' '
        'ON {} USING hnsw (vector vector_cosine_ops)'
    ))

