'''Offline-first diagnostics for the selected project.'''

from typing import Dict, List, Optional

import typer
from rich.table import Table

from osintgpt import Settings
from osintgpt.credentials import credential_status, resolve_credentials
from osintgpt.ingestion import Corpus
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    audit_locality,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm.registry import backend_spec
from osintgpt.projects import ProjectSettings, load_user_defaults
from osintgpt.vector_store import BACKENDS, store_for

from .output import emit, fail
from .selection import ProjectSelectionError, resolve_project, state_from


def _finding(findings: List[Dict[str, object]], name: str, ok: bool,
             detail: str) -> None:
    findings.append({'check': name, 'ok': ok, 'detail': detail})


def _provider_status(
    role: str,
    provider: str,
    model: str,
    settings: Settings,
    check_provider: bool,
    findings: List[Dict[str, object]]
) -> Dict[str, object]:
    backends = EMBEDDING_BACKENDS if role == 'embedding' else GENERATION_BACKENDS
    try:
        spec = backend_spec(provider, backends, role)
    except ValueError as error:
        _finding(findings, f'{role} provider', False, str(error))
        return {
            'backend': provider, 'model': model, 'ready': False,
            'checked': False, 'problem': str(error)
        }

    problems = []
    if spec.settings_field and not getattr(settings, spec.settings_field):
        problems.append(f'missing {spec.settings_field}')
    chosen_model = model or (
        settings.openai_embedding_model
        if role == 'embedding' else settings.openai_gpt_model
    ) or spec.default_model or ''
    if not chosen_model:
        problems.append('no model configured')

    ready = not problems
    detail = 'ready to build' if ready else '; '.join(problems)
    _finding(findings, f'{role} provider', ready, detail)
    status: Dict[str, object] = {
        'backend': provider,
        'model': chosen_model,
        'ready': ready,
        'checked': False,
        'problem': None if ready else detail
    }
    if not check_provider or not ready:
        return status
    if not spec.discovers_models:
        status['check_note'] = 'provider does not support model discovery'
        return status

    try:
        builder = (
            build_embedding_provider
            if role == 'embedding' else build_generation_provider
        )
        client = builder(provider, settings, model=chosen_model)
        status['checked'] = True
        status['available_models'] = client.list_models()
        _finding(findings, f'{role} reachability', True, 'provider responded')
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        status['checked'] = True
        status['problem'] = str(error)
        _finding(findings, f'{role} reachability', False, str(error))

    return status


def _credential_sources(home, findings: List[Dict[str, object]]):
    rows = []
    for status in credential_status(home):
        rows.append({
            'provider': status.provider,
            'variable': status.variable,
            'source': status.source,
            'shadowed': status.shadowed
        })
        if status.shadowed:
            _finding(
                findings, f'{status.provider} credential', False,
                f'{status.variable} in the environment is used instead of '
                'the stored credential'
            )

    return rows


def _store_status(project, settings: Settings,
                  check_provider: bool,
                  findings: List[Dict[str, object]]) -> Dict[str, object]:
    backend = (project.settings.storage_backend or 'sqlite').strip().lower()
    local_store = backend == 'sqlite'
    exists = project.paths.store.is_file() if local_store else None
    status: Dict[str, object] = {
        'backend': backend,
        'exists': exists,
        'chunks': 0 if local_store and not exists else None,
        'documents': 0 if local_store and not exists else None,
        'models': [] if local_store and not exists else None
    }
    if backend not in BACKENDS:
        detail = (
            f'unknown storage backend {backend!r}; choose one of: '
            f'{", ".join(BACKENDS)}'
        )
        status['problem'] = detail
        _finding(findings, 'store', False, detail)
        return status
    if local_store and not exists:
        _finding(findings, 'store', False, 'store file does not exist')
        return status
    if not local_store and not check_provider:
        _finding(
            findings, 'store', True,
            'remote store not contacted without --check-providers'
        )
        return status

    store = None
    try:
        store = store_for(project, settings)
        status.update({
            'chunks': store.count(),
            'documents': len(store.refs()),
            'models': store.models()
        })
        _finding(findings, 'store', True, 'store opened successfully')
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        status['problem'] = str(error)
        _finding(findings, 'store', False, str(error))
    finally:
        close = getattr(store, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as error:  # noqa: BLE001 — diagnostic boundary
                status['close_problem'] = str(error)

    return status


def _source_status(project, findings: List[Dict[str, object]]):
    rows = []
    try:
        corpus = Corpus.load(project.paths.sources)
        for source in corpus:
            try:
                covered = source.resolve(project.paths.root)
                rows.append({
                    'path': source.path,
                    'files': len(covered),
                    'problem': None
                })
            except Exception as error:  # noqa: BLE001 — diagnostic boundary
                rows.append({
                    'path': source.path, 'files': None,
                    'problem': str(error)
                })
                _finding(findings, f'source {source.path}', False, str(error))
        _finding(findings, 'sources', True, f'{len(rows)} registered')
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        _finding(findings, 'sources', False, str(error))

    return rows


def _locality(settings, effective, findings):
    try:
        report = audit_locality(
            settings,
            effective.embedding_provider,
            effective.generation_provider,
            effective.embedding_model or None,
            effective.generation_model or None
        )
        return {
            'local': report.is_local,
            'summary': report.summary,
            'setup': report.setup,
            'providers': [vars(provider) for provider in report.providers]
        }
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        _finding(findings, 'locality', False, str(error))
        return {'local': None, 'summary': str(error), 'setup': [], 'providers': []}


def doctor(
    context: typer.Context,
    project_slug: Optional[str] = typer.Option(
        None, '--project', help='Project slug or id; overrides selection.'
    ),
    check_providers: bool = typer.Option(
        False, '--check-providers',
        help='Contact configured providers and remote storage.'
    ),
    strict: bool = typer.Option(
        False, '--strict', help='Exit non-zero when a finding is unhealthy.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    state = state_from(context)
    try:
        project = resolve_project(state.home, project_slug)
    except ProjectSelectionError as error:
        fail(str(error), json_output)

    findings: List[Dict[str, object]] = []
    try:
        defaults = load_user_defaults(state.home)
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        defaults = ProjectSettings()
        _finding(findings, 'user defaults', False, str(error))
    try:
        base = resolve_credentials(state.home)
    except Exception as error:  # noqa: BLE001 — diagnostic boundary
        base = Settings()
        _finding(findings, 'environment', False, str(error))

    effective = project.effective_settings(defaults)
    settings = project.settings_for(base, defaults)
    embedding = _provider_status(
        'embedding', effective.embedding_provider,
        effective.embedding_model, settings, check_providers, findings
    )
    generation = _provider_status(
        'generation', effective.generation_provider,
        effective.generation_model, settings, check_providers, findings
    )
    storage = _store_status(project, settings, check_providers, findings)
    models = storage.get('models') or []
    configured_model = str(embedding['model'])
    if models:
        matches = configured_model in models
        detail = (
            f'{configured_model} matches stored vectors' if matches else
            f'model mismatch: configured {configured_model!r}; store has {models}'
        )
        _finding(findings, 'embedding model', matches, detail)

    data = {
        'project': {
            'id': project.id, 'slug': project.slug,
            'name': project.name, 'path': str(project.paths.root)
        },
        'storage': storage,
        'providers': {'embedding': embedding, 'generation': generation},
        'locality': _locality(settings, effective, findings),
        'credentials': _credential_sources(state.home, findings),
        'sources': _source_status(project, findings),
        'findings': findings
    }

    def render(target) -> None:
        target.print(f'Doctor: {project.name}', style='bold')
        target.print(f'Project: {project.slug} ({project.paths.root})')
        target.print(
            f'Storage: {storage["backend"]}; exists={storage["exists"]}; '
            f'chunks={storage["chunks"]}; documents={storage["documents"]}'
        )
        target.print(f'Stored models: {storage["models"]}')
        provider_table = Table(title='Providers')
        provider_table.add_column('Role')
        provider_table.add_column('Backend')
        provider_table.add_column('Model')
        provider_table.add_column('Ready')
        for role, status in data['providers'].items():
            provider_table.add_row(
                role, str(status['backend']), str(status['model']),
                str(status['ready'])
            )
        target.print(provider_table)
        target.print(f'Locality: {data["locality"]["summary"]}')
        target.print('Credentials', style='bold')
        for row in data['credentials']:
            where = row['source'] or 'not set'
            target.print(f'{row["provider"]}: {where}')
        target.print('Sources', style='bold')
        if not data['sources']:
            target.print('None registered.')
        for source in data['sources']:
            target.print(f'{source["path"]}: {source["files"]} files')
        target.print('Findings', style='bold')
        for item in findings:
            marker = 'OK' if item['ok'] else 'ISSUE'
            target.print(f'{marker} {item["check"]}: {item["detail"]}')

    emit(data, json_output, render)
    if strict and any(not item['ok'] for item in findings):
        raise typer.Exit(code=1)
