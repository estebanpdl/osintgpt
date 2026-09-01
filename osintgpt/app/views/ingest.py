# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: ingest.py
# Description: Registering material and indexing it. What it would cost is
#   shown before anything is spent.
# =================================================================================

# type hints
from typing import Any, Dict, List

# import osintgpt
from osintgpt import index_project

# import osintgpt ingestion
from osintgpt.ingestion import Corpus, FieldMapping, describe_fields, dry_run
from osintgpt.ingestion.loaders import needs_mapping
from osintgpt.ingestion.transcription import transcriber_for_project

from ..browse import directory_input


# what a folder would contribute
def preview(folder) -> Dict[str, Any]:
    '''
    Read a folder without embedding anything.

    Args:
        folder (Path): Directory to preview.

    Returns:
        Dict[str, Any]: The summary, plus files still needing field roles.
    '''
    run = dry_run(folder)

    return {
        'summary': run.summary,
        'documents': run.documents,
        'chunks': run.chunks,
        'tokens': run.tokens,
        'cost': run.estimated_cost,
        'vision_pages': run.vision_pages,
        'unconfigured': [f.path for f in run.unconfigured],
        'failed': [(f.path, f.problem) for f in run.failed]
    }


# what a structured file needs before it can be indexed
def field_roles(path) -> List[str]:
    '''
    Args:
        path (Path): A structured file.

    Returns:
        List[str]: Its field names, empty when the file needs no mapping. \
            The analyst chooses which carry content; nothing here guesses, \
            and a wrong guess would embed identifiers as if they were prose.
    '''
    if not needs_mapping(path):
        return []

    try:
        return list(describe_fields(path))
    except Exception:  # noqa: BLE001 — a view, not a pass
        return []


# render the ingest view
def render(st, runtime, state) -> None:
    '''
    Args:
        st: The Streamlit module.
        runtime (Runtime): Project and providers.
        state: Session state.
    '''
    from pathlib import Path

    project = runtime.project
    st.subheader(f'Material — {project.name}')

    folder = directory_input(
        st, 'Folder to register', 'ingest-folder', state,
        help_text='A directory of documents. Everything readable beneath it '
                  'is tracked, including files added later.'
    )
    if folder:
        if Path(folder).is_dir():
            _register(st, project, Path(folder))
        else:
            st.error(f'{folder} is not a directory.')

    corpus = Corpus.load(project.paths.sources)
    if len(corpus):
        st.caption('Registered')
        for source in corpus:
            covered = len(source.resolve(project.paths.root))
            st.text(f'{source.path} — {covered} files')
    else:
        st.info('Nothing registered yet.')

        return

    _preview(st, project)

    # Behind a button, never on page load: every widget interaction reruns
    # this script, and an indexing pass triggered by rendering runs again on
    # each one.
    if st.button('Index now', type='primary'):
        _index(st, runtime)


def _register(st, project, folder) -> None:
    from pathlib import Path

    unmapped = [
        path for path in folder.rglob('*')
        if path.is_file() and field_roles(path)
    ]

    mapping = None
    if unmapped:
        st.warning(
            f'{len(unmapped)} structured file(s) need you to say which fields '
            'carry content before they can be indexed.'
        )
        sample = unmapped[0]
        fields = field_roles(sample)
        chosen = st.multiselect(
            f'Content fields in {sample.name}', fields, key='content-fields'
        )
        if chosen:
            mapping = FieldMapping(content=tuple(chosen))

    if st.button('Register this folder'):
        Corpus.load(project.paths.sources).register(
            str(folder), mapping
        )
        st.rerun()


def _preview(st, project) -> None:
    if not st.checkbox('Preview what would be indexed'):
        return

    with st.spinner('Reading…'):
        facts = preview(project.paths.root)

    st.text(facts['summary'])
    if facts['vision_pages']:
        st.warning(
            f'{facts["vision_pages"]} PDF page(s) would need a vision model, '
            'at one generation call each.'
        )
    for path, problem in facts['failed']:
        st.error(f'{path.name}: {problem}')


def _index(st, runtime) -> None:
    progress = st.progress(0.0)
    status = st.empty()

    def report(ref, position, total):
        progress.progress(position / max(total, 1))
        status.text(f'{position}/{total}  {ref}')

    # The generator is built only if a scanned page is actually found, so a
    # corpus of born-digital documents never needs a generation credential.
    report_result = index_project(
        runtime.project, runtime.embedder, on_progress=report,
        transcriber=transcriber_for_project(
            runtime.project, lambda: runtime.generator
        )
    )
    progress.empty()
    status.empty()

    st.success(report_result.summary)
    for failure in report_result.failed:
        st.error(f'{failure.ref}: {failure.problem}')
    for notice in report_result.notices:
        st.warning(notice)
