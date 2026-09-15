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

# import submodules
from pathlib import Path

# type hints
from typing import Any, Dict

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL

# import osintgpt ingestion
from osintgpt.ingestion import Corpus, preview_corpus, to_markdown
from osintgpt.ingestion.transcription import transcriber_for

from ..browse import directory_input
from .mapping import assign_roles, described_fields, schema_groups
from . import index_progress

# Registering ends in a rerun, which discards what the current pass drew, so
# anything the operator has to read about it survives into the next one.
NOTICE = 'ingest-notice'

# The last conversion, held because downloading it reruns the script and the
# button that produced it would then read as unpressed.
RESULT = 'convert-result'
PROBLEM = 'convert-problem'


# what the registered corpus would contribute
def preview(corpus, root, embedding_model: str = '') -> Dict[str, Any]:
    '''
    Read the registered sources without embedding anything.

    Args:
        corpus (Corpus): The project's registered sources.
        root (Path): Project root, which project-local sources are relative to.
        embedding_model (str): Model whose tokenizer and price the estimate \
            uses. Empty falls back to the library default, whose price is \
            not this project's — encodings and prices both differ by model.

    Returns:
        Dict[str, Any]: The summary, plus files still needing field roles.
    '''
    run = preview_corpus(
        corpus, root, embedding_model=embedding_model or DEFAULT_EMBEDDING_MODEL
    )

    return {
        'summary': run.summary,
        'documents': run.documents,
        'chunks': run.chunks,
        'tokens': run.tokens,
        'cost': run.estimated_cost,
        'vision_pages': run.vision_pages,
        'record_counts': [{'Source': str(f.path), **f.statistics} for f in run.files if f.statistics],
        'unconfigured': [f.path for f in run.unconfigured],
        'failed': [(f.path, f.problem) for f in run.failed]
    }


# render the ingest view
def render(st, runtime, state) -> None:
    '''
    Args:
        st: The Streamlit module.
        runtime (Runtime): Project and providers.
        state: Session state.
    '''
    project = runtime.project
    st.subheader(f'Material — {project.name}')

    notice = state.pop(NOTICE, '')
    if notice:
        st.warning(notice)

    _convert(st, runtime, state)

    folder = directory_input(
        st, 'Folder to register', 'ingest-folder', state,
        help_text='A directory of documents. Everything readable beneath it '
                  'is tracked, including files added later.'
    )
    if folder:
        if Path(folder).is_dir():
            _register(st, project, Path(folder), state)
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

    _preview(st, runtime, corpus)

    # Behind a button, never on page load: every widget interaction reruns
    # this script, and an indexing pass triggered by rendering runs again on
    # each one.
    index_progress.render(st, runtime, state)


def _convert(st, runtime, state) -> None:
    '''
    Read one file the way indexing would and show the markdown. Nothing is
    registered, embedded or stored, and no model is contacted unless the
    operator asks for one.
    '''
    with st.expander('Convert a document'):
        typed = st.text_input(
            'File to convert', key='convert-path',
            help='Any readable file. The markdown shown is what chunking '
                 'would be given.'
        ).strip()
        transcribe = st.checkbox(
            'Transcribe scanned pages', key='convert-vision',
            help='One generation call per page holding no extractable text. '
                 'Left off, nothing leaves this machine.'
        )

        path = Path(typed) if typed else None
        mapping = None
        if path is not None and path.is_file():
            described = described_fields(path)
            if described:
                mapping = assign_roles(st, [path], described, 'convert-map')

        if st.button('Convert', key='convert-run'):
            _run_conversion(st, runtime, state, path, mapping, transcribe)

        _show_conversion(st, state, path)


def _run_conversion(st, runtime, state, path, mapping, transcribe) -> None:
    state.pop(RESULT, None)
    state.pop(PROBLEM, None)

    if path is None or not path.is_file():
        state[PROBLEM] = 'Give the path to a file.'

        return

    transcriber = transcriber_for(
        runtime.generator, runtime.project.paths.extracts
    ) if transcribe else None

    try:
        with st.spinner(f'Reading {path.name}…'):
            state[RESULT] = to_markdown(path, mapping, transcriber)
    except Exception as error:  # noqa: BLE001 — one file, reported in place
        state[PROBLEM] = str(error)


def _show_conversion(st, state, path) -> None:
    problem = state.get(PROBLEM, '')
    if problem:
        st.error(problem)

    conversion = state.get(RESULT)
    # A result stays on screen across the reruns a download or a checkbox
    # causes, but not after the path changes under it.
    if conversion is None or (path is not None and path != conversion.path):
        return

    detail = (
        f'Read by {conversion.reader} — {conversion.documents} document(s)'
    )
    if conversion.pages:
        detail += f', {conversion.pages} page(s)'
    st.caption(detail)

    if conversion.empty_pages:
        st.warning(
            f'{conversion.empty_pages} page(s) hold no extractable text. '
            'Transcribing them costs one generation call each.'
        )

    st.download_button(
        'Download markdown', conversion.markdown,
        file_name=f'{conversion.path.name}.md', mime='text/markdown',
        key='convert-download'
    )
    st.code(conversion.markdown, language='markdown', wrap_lines=True)


def _register(st, project, folder, state) -> None:
    groups = schema_groups(folder)
    if groups:
        st.warning(
            f'{sum(len(paths) for paths, _ in groups)} structured file(s) need '
            'you to say which fields carry content before they can be indexed.'
        )

    mappings = [
        (paths, assign_roles(st, paths, described, f'map-{position}'))
        for position, (paths, described) in enumerate(groups)
    ]

    if not st.button('Register this folder'):
        return

    corpus = Corpus.load(project.paths.sources)
    # The folder carries the prose; each structured file is registered on its
    # own because a mapping belongs to a schema, not to a directory.
    # `Corpus.mappings` lets a file's own registration win over its folder.
    corpus.register(str(folder))
    mapped = 0
    for paths, mapping in mappings:
        if mapping is None:
            continue
        for path in paths:
            corpus.register(path, mapping)
            mapped += 1

    unmapped = sum(len(paths) for paths, _ in groups) - mapped
    if unmapped:
        state[NOTICE] = (
            f'{unmapped} structured file(s) were left without a content '
            'field and will be skipped until one is named.'
        )
    st.rerun()


def _preview(st, runtime, corpus) -> None:
    if not st.checkbox('Preview what would be indexed'):
        return

    with st.spinner('Reading…'):
        facts = preview(
            corpus, runtime.project.paths.root, runtime.embedding_model
        )

    st.text(facts['summary'])
    st.caption('Full registered corpus preview, including material already indexed or checkpointed. This is not an estimate of the remaining run cost.')
    if facts['record_counts']:
        st.dataframe(facts['record_counts'], hide_index=True, use_container_width=True)
    if facts['vision_pages']:
        st.warning(
            f'{facts["vision_pages"]} PDF page(s) would need a vision model, '
            'at one generation call each.'
        )
    for path, problem in facts['failed']:
        st.error(f'{path.name}: {problem}')
