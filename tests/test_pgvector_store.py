# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_pgvector_store.py
# Description: What is true of the Postgres store and not of every store —
#   including that it stays optional, which is the part with no server needed
#   to check.
# =================================================================================

# import modules
import pytest

# import osintgpt config
from osintgpt.config import Settings

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

from osintgpt.vector_store import BACKENDS
from osintgpt.vector_store.pgvector_store import (
    TABLE_PREFIX,
    PgVectorStore,
    _identifier,
    _vector_literal
)


class TestItStaysOptional:
    '''
    The plan's requirement, and the half that needs no server: a project on
    the default store must never need these drivers, and no error may tell an
    operator to go install Postgres to use osintgpt.
    '''

    def test_importing_the_package_does_not_import_the_drivers(self):
        import osintgpt.vector_store as package

        assert not hasattr(package, 'PgVectorStore')

    def test_the_factory_imports_it_only_when_asked(self):
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent
            / 'osintgpt' / 'vector_store' / 'factory.py'
        ).read_text(encoding='utf-8')
        tree = ast.parse(source)

        top_level = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        names = [
            alias.name for node in top_level
            for alias in getattr(node, 'names', [])
        ]

        assert not any('pgvector' in name for name in names)

    def test_postgres_is_a_named_backend(self):
        assert 'postgres' in BACKENDS

    def test_it_is_not_the_default(self):
        from osintgpt.projects import ProjectSettings

        assert ProjectSettings().storage_backend == 'sqlite'

    def test_a_missing_dsn_says_what_is_missing_not_what_to_install(self):
        pytest.importorskip('psycopg')

        with pytest.raises(MissingEnvironmentVariableError) as excinfo:
            PgVectorStore(Settings())

        message = str(excinfo.value)

        assert 'POSTGRES_DSN' in message
        assert 'install' not in message.lower()


class TestTableNaming:
    '''
    One table per project rather than a project column, so a forgotten WHERE
    cannot leak one case into another.
    '''

    @pytest.mark.parametrize('slug, expected', [
        ('case-alpha', 'case_alpha'),
        ('Case Alpha', 'case_alpha'),
        ('case.alpha', 'case_alpha'),
        ('--leading', 'leading'),
        ('trailing--', 'trailing')
    ])
    def test_a_slug_becomes_a_readable_identifier(self, slug, expected):
        assert _identifier(slug) == expected

    def test_a_slug_with_nothing_usable_still_names_something(self):
        assert _identifier('---') == 'default'

    def test_non_ascii_slugs_do_not_become_empty(self):
        '''
        Project names are not English. A slug of accented or non-Latin
        characters must still produce a table, not collapse to the fallback.
        '''
        assert _identifier('análisis') == 'análisis'
        assert _identifier('анализ') == 'анализ'

    def test_tables_are_prefixed(self):
        pytest.importorskip('psycopg')

        assert TABLE_PREFIX.startswith('osintgpt')


class TestVectorLiteral:
    def test_it_writes_pgvector_text_form(self):
        assert _vector_literal([1.0, 0.0, -0.5]) == '[1.0,0.0,-0.5]'

    def test_it_accepts_integers(self):
        assert _vector_literal([1, 0]) == '[1.0,0.0]'

    def test_an_empty_vector_is_still_valid_syntax(self):
        assert _vector_literal([]) == '[]'
