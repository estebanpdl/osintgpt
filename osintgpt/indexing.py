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
from bisect import bisect_right
from osintgpt.index_events import emit_index, observe_index

# type hints
from typing import Callable, List, Optional, Union

# import osintgpt ingestion
from osintgpt.ingestion import (
    Corpus,
    IndexState,
    is_image,
    marker_for,
    read_image
)
from osintgpt.ingestion.images import NO_IMAGE_SUPPORT
from osintgpt.ingestion.checkpoints import IndexJournal, fingerprint
from osintgpt.ingestion.indexing import content_hash
from osintgpt.ingestion.recipes import (
    destination_for, prepare_chunks, recipe_for, reconcile_plan, supports_legacy_checkpoint
)

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider, EmbeddingPurpose
from osintgpt.llm.usage import CostLimitReached

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
    adopted: int = 0

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
            parts.append(f'{len(self.failed)} failed')
        if self.adopted:
            parts.append(f'{self.adopted} legacy configurations adopted')

        return ', '.join(parts) if parts else 'nothing to do'

    @property
    def notices(self) -> List[str]:
        '''
        Returns:
            List[str]: Why each skipped file was skipped. A file that cannot                 be indexed is reported rather than dropped, so an operator                 learns it from a run instead of from an answer that was                 missing something.
        '''
        notices = [result.problem for result in self.skipped]
        if self.adopted:
            notices.append(
                'Legacy chunks matched the source, mapping and model. Adopted '
                'the standard OpenAI endpoint and default embedding options; '
                'legacy indexes did not record these settings. No embeddings were requested.'
            )
        return notices


# index a project
def index_project(
    project: Project,
    embedder: EmbeddingProvider,
    store: Optional[BaseVectorEngine] = None,
    force: bool = False,
    purge_other_models: bool = False,
    on_progress: Optional[Progress] = None,
    config=None,
    transcriber=None,
    on_event=None
) -> IndexReport:
    '''
    Bring a project's index up to date with its registered corpus.

    Only documents that are new or changed are read and embedded; the rest
    cost a hash and a destination inventory check. Documents no source covers any more have their chunks
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
        on_event (Callable, optional): Receives run-local IndexEvent values
            for chunks, durable checkpoints, provider waits/retries and usage.
        config (Settings, optional): Connection settings, needed only by a \
            backend that reaches a server.

        transcriber (Callable, optional): Reads PDF pages holding no \
            extractable text. Without one those pages are indexed as a named \
            gap rather than failing, so a corpus of born-digital documents \
            needs no vision model at all.

    Returns:
        IndexReport: What the pass did, per document.
    '''
    with observe_index(on_event), IndexJournal(project.paths.index_journal) as journal:
        emit_index('planning', model=embedder.model, provider=_provider_name(embedder))
        owned = store is None
        store = store or store_for(project, config)
        try:
            return _run(
                project, embedder, store, force, purge_other_models, on_progress,
                transcriber, journal
            )
        finally:
            # A supplied store belongs to its caller. The lock also covers
            # opening/migrating a store so competing app/CLI jobs cannot race.
            if owned and hasattr(store, 'close'):
                store.close()


def _run(
    project: Project,
    embedder: EmbeddingProvider,
    store: BaseVectorEngine,
    force: bool,
    purge_other_models: bool,
    on_progress: Optional[Progress],
    transcriber=None,
    journal=None
) -> IndexReport:
    root = project.paths.root
    corpus = Corpus.load(project.paths.sources)
    state = IndexState.load(project.paths.index_state)

    files = corpus.files(root)
    mappings = {_ref_for(path, root): corpus.mapping_for(path, root) for path in files}
    recipes = {
        _ref_for(path, root): recipe_for(
            mappings[_ref_for(path, root)], embedder,
            transcribes=path.suffix.lower() == '.pdf' and transcriber is not None
        ) for path in files
    }
    destination = destination_for(store, root)
    plan = state.plan(files, root, force=force, recipes=recipes, destination=destination)
    pending = journal.pending_refs()
    plan, adopted, blocked = reconcile_plan(
        plan, state, root, embedder, store, mappings, recipes, destination, pending
    )
    indexed: List[DocumentResult] = []
    failed = [DocumentResult(ref=ref, problem=problem) for ref, problem in blocked]
    skipped: List[DocumentResult] = []
    emit_index('plan', files=len(files), work=len(plan.work), unchanged=len(plan.unchanged), blocked=len(blocked))
    for ref, problem in blocked:
        emit_index('failed', ref=ref, problem=problem)

    for position, path in enumerate(plan.work, 1):
        ref = _ref_for(path, root)
        emit_index('reading', ref=ref, position=position, total=len(plan.work))
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
            emit_index('skipped', ref=ref, problem=notice)
            continue

        statistics = {}
        try:
            source_hash = content_hash(path.read_bytes())
            stored = _index_document(
                path, ref, mappings[ref], embedder, store, transcriber,
                journal, force, source_hash, recipes[ref], destination, statistics
            )
        except CostLimitReached as error:
            state.save()
            emit_index('stopped', ref=ref, problem=str(error))
            raise error.with_index_progress(
                len(indexed), len(plan.work) - position + 1
            )
        except Exception as error:  # noqa: BLE001 — one document, not the pass
            log.warning('%s could not be indexed: %s', ref, error)
            failed.append(DocumentResult(ref=ref, problem=str(error)))
            emit_index('failed', ref=ref, problem=str(error))
            continue

        state.record(
            path, root, chunks=stored, source_hash=source_hash,
            recipe=recipes[ref], destination=destination, embedding_model=embedder.model,
            embedding_provider=_provider_name(embedder), statistics=statistics
        )
        # The job remains recoverable until BOTH the store and index state
        # are saved. Re-publication after a crash reuses the saved vectors.
        state.save()
        journal.forget(ref)
        indexed.append(DocumentResult(ref=ref, chunks=stored))
        emit_index('published', ref=ref, chunks=stored, statistics=dict(statistics))

    removed = 0
    if plan.removed:
        removed = store.delete(plan.removed)
        state.forget(plan.removed)
    for ref in pending - {_ref_for(path, root) for path in files}:
        journal.forget(ref)

    purged = 0
    if purge_other_models and not failed and not skipped:
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
        embedding_model=embedder.model,
        adopted=len(adopted)
    )


def _index_document(
    path: Path,
    ref: str,
    mapping,
    embedder: EmbeddingProvider,
    store: BaseVectorEngine,
    transcriber=None,
    journal=None,
    force=False,
    source_hash=None,
    recipe=None,
    destination=None,
    statistics=None
) -> int:
    '''
    Read, chunk, embed and store one document. Returns the chunks stored.
    '''
    if is_image(path):
        if statistics is not None:
            statistics.update(documents=1, chunked_documents=1)
        emit_index('embedding', ref=ref, total=1, completed=0, completed_records=0, statistics=dict(statistics or {}))
        # One image is one chunk: there is nothing to split, and the stored
        # text is a marker rather than a caption, because nothing was
        # extracted and inventing a description would put words in the index
        # that no model produced.
        vector = embedder.embed_images(
            [read_image(path)], purpose=EmbeddingPurpose.DOCUMENT
        )[0]

        if source_hash != content_hash(path.read_bytes()):
            raise RuntimeError('Source changed while indexing; re-run index to use its new content.')

        return store.upsert(
            ref,
            [StoredChunk(
                ref=ref,
                sequence=0,
                text=marker_for(path),
                embedding_model=embedder.model,
                document_ref=ref
            )],
            [vector]
        )

    record_ends = []
    chunks, texts, document_refs = prepare_chunks(
        path, ref, mapping, embedder, transcriber, statistics=statistics, record_ends=record_ends
    )
    details = {
        'provider': _provider_name(embedder), 'model': embedder.model,
        'recipe': recipe, 'destination': destination, 'source_hash': source_hash,
        'statistics': dict(statistics or {})
    }

    signature = fingerprint(
        source_hash, mapping, embedder, chunks, texts, document_refs,
        recipe=recipe, destination=destination
    )
    # The first checkpoint format already hashed exact input plus provider,
    # model, endpoint and task style. Adopt that prefix without repeating calls.
    legacy_signature = fingerprint(
        source_hash, mapping, embedder, chunks, texts, document_refs
    )
    completed = journal.prepare(
        ref, signature, len(chunks), reset=force,
        compatible_signatures=(legacy_signature,) if supports_legacy_checkpoint(embedder) else (),
        details=details
    )
    emit_index('embedding', ref=ref, total=len(chunks), completed=completed,
               completed_records=bisect_right(record_ends, completed), statistics=dict(statistics or {}))

    def checkpoint(vectors):
        nonlocal completed
        end = completed + len(vectors)
        journal.save_batch(
            ref, completed, chunks[completed:end], document_refs[completed:end], vectors,
            details={**details, 'completed_records': bisect_right(record_ends, end)}
        )
        completed = end
        emit_index('checkpoint', ref=ref, total=len(chunks), completed=completed,
                   completed_records=bisect_right(record_ends, completed), statistics=dict(statistics or {}))

    if completed < len(texts):
        # Duck-typed library providers predating checkpoints still work. The
        # built-in paid providers deliver one response at a time to this sink.
        checkpointed = getattr(embedder, 'embed_checkpointed', None)
        if checkpointed is None:
            checkpoint(embedder.embed(texts[completed:], purpose=EmbeddingPurpose.DOCUMENT))
        else:
            checkpointed(
                texts[completed:], purpose=EmbeddingPurpose.DOCUMENT, on_batch=checkpoint
            )

    if source_hash != content_hash(path.read_bytes()):
        raise RuntimeError('Source changed while indexing; re-run index to use its new content.')
    emit_index('publishing', ref=ref, total=len(chunks), completed=completed)
    return store.upsert(ref, chunks, journal.vectors_for(ref))


def _provider_name(embedder):
    # Display metadata only; this must not change an existing recipe hash.
    return getattr(embedder, 'provider', '') or ('gemini' if type(embedder).__name__ == 'GeminiEmbedding' else '')


def _ref_for(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
