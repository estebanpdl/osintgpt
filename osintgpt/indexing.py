# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: indexing.py
# Description: Turning a project's registered corpus into searchable vectors.
#   The one place ingestion, embedding and storage meet.
# =================================================================================

# import modules
import logging

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Callable, List, Optional, Union

# import osintgpt ingestion
from osintgpt.ingestion import (
    Corpus,
    IndexState,
    chunk_document,
    is_image,
    load_documents,
    marker_for,
    read_image
)
from osintgpt.ingestion.images import NO_IMAGE_SUPPORT

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider

# import osintgpt projects
from osintgpt.projects import Project

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, StoredChunk, store_for

log = logging.getLogger('osintgpt.indexing')

# Reports a document as it is handled, so a long pass says what it is doing
# rather than going quiet. Called with the ref and the running position.
Progress = Callable[[str, int, int], None]


# DocumentResult class
@dataclass(frozen=True)
class DocumentResult:
    '''
    What one document contributed, or why it contributed nothing.
    '''
    ref: str
    chunks: int = 0
    # Set when the document could not be read. One bad file does not stop a
    # pass, so the failure is carried rather than raised.
    problem: str = ''

    @property
    def ok(self) -> bool:
        return not self.problem


# IndexReport class
@dataclass(frozen=True)
class IndexReport:
    '''
    What a pass did.
    '''
    indexed: List[DocumentResult] = field(default_factory=list)
    failed: List[DocumentResult] = field(default_factory=list)
    # Files a pass declined to index, each with the reason. Separate from
    # failures: nothing went wrong, the configuration cannot hold them.
    skipped: List[DocumentResult] = field(default_factory=list)
    unchanged: int = 0
    removed: int = 0
    # Chunks dropped because they were embedded by a model no longer in use.
    purged: int = 0
    embedding_model: str = ''

    @property
    def chunks(self) -> int:
        return sum(result.chunks for result in self.indexed)

    @property
    def summary(self) -> str:
        parts = []
        if self.indexed:
            parts.append(f'{len(self.indexed)} documents, {self.chunks} chunks')
        if self.unchanged:
            parts.append(f'{self.unchanged} unchanged')
        if self.removed:
            parts.append(f'{self.removed} removed')
        if self.purged:
            parts.append(f'{self.purged} chunks purged from other models')
        if self.skipped:
            parts.append(f'{len(self.skipped)} skipped')
        if self.failed:
            parts.append(f'{len(self.failed)} unreadable')

        return ', '.join(parts) if parts else 'nothing to do'

    @property
    def notices(self) -> List[str]:
        '''
        Returns:
            List[str]: Why each skipped file was skipped. A file that cannot                 be indexed is reported rather than dropped, so an operator                 learns it from a run instead of from an answer that was                 missing something.
        '''
        return [result.problem for result in self.skipped]


# index a project
def index_project(
    project: Project,
    embedder: EmbeddingProvider,
    store: Optional[BaseVectorEngine] = None,
    force: bool = False,
    purge_other_models: bool = False,
    on_progress: Optional[Progress] = None,
    config=None,
    transcriber=None
) -> IndexReport:
    '''
    Bring a project's index up to date with its registered corpus.

    Only documents that are new or changed are read and embedded; the rest
    cost a hash. Documents no source covers any more have their chunks
    removed, because vectors nobody covers are invisible to a corpus walk and
    still returned by a search.

    Args:
        project (Project): The project to index.
        embedder (EmbeddingProvider): Produces the vectors. Its `model` is \
            stored with every chunk and filtered on every search.
        store (BaseVectorEngine, optional): Where vectors go. Defaults to \
            whichever backend the project's settings name.
        force (bool): Re-embed everything. The escape hatch for a chunker \
            change, which alters output without altering any document.
        purge_other_models (bool): Drop chunks left by a previous embedding \
            model. Runs only after the re-embed succeeds, so a failed pass \
            costs nothing.
        on_progress (Progress, optional): Called per document with the ref, \
            its position and the total.
        config (Settings, optional): Connection settings, needed only by a \
            backend that reaches a server.

        transcriber (Callable, optional): Reads PDF pages holding no \
            extractable text. Without one those pages are indexed as a named \
            gap rather than failing, so a corpus of born-digital documents \
            needs no vision model at all.

    Returns:
        IndexReport: What the pass did, per document.
    '''
    owned = store is None
    store = store or store_for(project, config)

    try:
        return _run(
            project, embedder, store, force, purge_other_models, on_progress,
            transcriber
        )
    finally:
        # A store this function opened is a store it closes; one passed in
        # belongs to the caller.
        if owned and hasattr(store, 'close'):
            store.close()


def _run(
    project: Project,
    embedder: EmbeddingProvider,
    store: BaseVectorEngine,
    force: bool,
    purge_other_models: bool,
    on_progress: Optional[Progress],
    transcriber=None
) -> IndexReport:
    root = project.paths.root
    corpus = Corpus.load(project.paths.sources)
    state = IndexState.load(project.paths.index_state)

    plan = state.plan(corpus.files(root), root, force=force)
    indexed: List[DocumentResult] = []
    failed: List[DocumentResult] = []
    skipped: List[DocumentResult] = []

    for position, path in enumerate(plan.work, 1):
        ref = _ref_for(path, root)
        if on_progress:
            on_progress(ref, position, len(plan.work))

        if is_image(path) and not embedder.supports_images:
            # Declined, not failed, and recorded either way. A registered file
            # that quietly never reaches the index is the worst outcome here:
            # the corpus looks complete and answers are missing a source.
            notice = NO_IMAGE_SUPPORT.format(
                name=ref, model=embedder.model
            )
            log.info('%s', notice)
            skipped.append(DocumentResult(ref=ref, problem=notice))
            continue

        try:
            stored = _index_document(
                path, ref, root, corpus, embedder, store, transcriber
            )
        except Exception as error:  # noqa: BLE001 — one document, not the pass
            log.warning('%s could not be indexed: %s', ref, error)
            failed.append(DocumentResult(ref=ref, problem=str(error)))
            continue

        state.record(path, root, chunks=stored)
        indexed.append(DocumentResult(ref=ref, chunks=stored))

    removed = 0
    if plan.removed:
        removed = store.delete(plan.removed)
        state.forget(plan.removed)

    purged = 0
    if purge_other_models:
        # After the re-embed, never before: a failed pass would otherwise
        # leave a project with neither the old vectors nor the new ones.
        purged = store.purge_other_models(keep=embedder.model)

    state.save()

    return IndexReport(
        indexed=indexed,
        failed=failed,
        skipped=skipped,
        unchanged=len(plan.unchanged),
        removed=removed,
        purged=purged,
        embedding_model=embedder.model
    )


def _index_document(
    path: Path,
    ref: str,
    root: Path,
    corpus: Corpus,
    embedder: EmbeddingProvider,
    store: BaseVectorEngine,
    transcriber=None
) -> int:
    '''
    Read, chunk, embed and store one document. Returns the chunks stored.
    '''
    if is_image(path):
        # One image is one chunk: there is nothing to split, and the stored
        # text is a marker rather than a caption, because nothing was
        # extracted and inventing a description would put words in the index
        # that no model produced.
        vector = embedder.embed_images([read_image(path)])[0]

        return store.upsert(
            ref,
            [StoredChunk(
                ref=ref,
                sequence=0,
                text=marker_for(path),
                embedding_model=embedder.model
            )],
            [vector]
        )

    documents = load_documents(
        path, corpus.mapping_for(path, root), transcriber
    )

    chunks: List[StoredChunk] = []
    texts: List[str] = []

    for document in documents:
        for piece in chunk_document(document.text):
            # The rendered form carries the section path, so what is embedded
            # is what a reader would see. The path is kept separately too, so
            # a citation does not have to parse it back out.
            texts.append(piece.rendered)
            chunks.append(StoredChunk(
                ref=ref,
                sequence=len(chunks),
                text=piece.text,
                embedding_model=embedder.model,
                path=piece.path,
                timestamp=document.timestamp,
                author=document.author,
                metadata=dict(document.metadata)
            ))

    if not chunks:
        # A document that produces nothing still replaces what it had: an
        # emptied file should not keep answering searches.
        store.upsert(ref, [], [])

        return 0

    return store.upsert(ref, chunks, embedder.embed(texts))


def _ref_for(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
