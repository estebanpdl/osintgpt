# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: build.py
# Description: Building a project's graph. One generation call per document,
#   so it happens because someone asked for it and never as a side effect.
# =================================================================================

# import modules
import logging

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Callable, List, Optional

# import osintgpt ingestion
from osintgpt.ingestion import Corpus, is_image, load_documents

# import osintgpt llm
from osintgpt.llm.base import GenerationProvider

# import osintgpt projects
from osintgpt.projects import Project

from .extraction import Extraction, extract_document
from .store import GraphStore, graph_for

log = logging.getLogger('osintgpt.graph')

# Reports a document as it is read. A pass costing one call per document
# should say what it is doing rather than going quiet.
Progress = Callable[[str, int, int], None]

NOT_ENABLED = (
    '{slug}: the graph is off for this project. It costs one generation call '
    'per document, so it is built when asked for and never as a side effect. '
    'Enable it with graph_enabled, then build it.'
)

NOT_BUILT = (
    '{slug}: the graph has not been built yet, so there is nothing to update. '
    'Build it once explicitly first.'
)


# GraphReport class
@dataclass(frozen=True)
class GraphReport:
    '''
    What a build pass did.
    '''
    extracted: List[Extraction] = field(default_factory=list)
    failed: List[Extraction] = field(default_factory=list)
    skipped: int = 0
    removed: int = 0
    # Set when the pass declined to run at all, with the reason.
    refused: str = ''

    @property
    def entities(self) -> int:
        return sum(len(e.entities) for e in self.extracted)

    @property
    def edges(self) -> int:
        return sum(len(e.edges) for e in self.extracted)

    @property
    def calls(self) -> int:
        '''
        Returns:
            int: Generation calls this pass made — the number that decides \
                what it cost.
        '''
        return len(self.extracted) + len(self.failed)

    @property
    def summary(self) -> str:
        if self.refused:
            return self.refused

        parts = []
        if self.extracted:
            parts.append(
                f'{len(self.extracted)} documents, {self.entities} entities, '
                f'{self.edges} edges'
            )
        if self.skipped:
            parts.append(f'{self.skipped} unchanged')
        if self.removed:
            parts.append(f'{self.removed} edges removed')
        if self.failed:
            parts.append(f'{len(self.failed)} failed')

        return ', '.join(parts) if parts else 'nothing to do'


# build or update a project's graph
def build_graph(
    project: Project,
    generator: GenerationProvider,
    incremental: bool = False,
    rebuild: bool = False,
    store: Optional[GraphStore] = None,
    on_progress: Optional[Progress] = None
) -> GraphReport:
    '''
    Extract entities and relationships from the project's registered corpus.

    One generation call per document, which is why nothing calls this to
    answer a question. A project with the graph disabled refuses, and an
    incremental pass over a graph that was never built refuses too rather than
    quietly becoming the first build.

    Args:
        project (Project): The project to build for.
        generator (GenerationProvider): Reads the documents.
        incremental (bool): Only documents the graph has not seen. Requires \
            an existing graph.
        rebuild (bool): Forget what a document asserted before re-reading it, \
            for a corpus whose documents have changed.
        store (GraphStore, optional): Defaults to the project's own.
        on_progress (Progress, optional): Called per document with the ref, \
            its position and the total.

    Returns:
        GraphReport: What the pass did, or why it declined.
    '''
    if not project.settings.graph_enabled:
        return GraphReport(refused=NOT_ENABLED.format(slug=project.slug))

    owned = store is None
    graph = store or graph_for(project)

    try:
        if incremental and not graph.is_built:
            # Without this, "keep it current" would become "build it", and an
            # operator would pay for a corpus-wide pass they never chose.
            return GraphReport(refused=NOT_BUILT.format(slug=project.slug))

        return _run(project, generator, graph, incremental, rebuild,
                    on_progress)
    finally:
        if owned:
            graph.close()


def _run(project, generator, graph, incremental, rebuild, on_progress):
    root = project.paths.root
    corpus = Corpus.load(project.paths.sources)
    already = set(graph.refs())

    targets = []
    skipped = 0
    for path in corpus.files(root):
        if is_image(path):
            # An image has no text to read relationships out of.
            continue
        ref = _ref_for(path, root)
        if incremental and ref in already:
            skipped += 1
            continue
        targets.append((ref, path))

    extracted: List[Extraction] = []
    failed: List[Extraction] = []
    removed = 0

    for position, (ref, path) in enumerate(targets, 1):
        if on_progress:
            on_progress(ref, position, len(targets))

        try:
            documents = load_documents(path, corpus.mapping_for(path, root))
        except Exception as error:  # noqa: BLE001 — one document, not the pass
            log.warning('%s could not be read: %s', ref, error)
            failed.append(Extraction(ref=ref, problem=str(error)))
            continue

        text = '\n\n'.join(d.text for d in documents if d.text).strip()
        if not text:
            continue

        result = extract_document(generator, ref, text)
        if not result.ok:
            failed.append(result)
            continue

        if rebuild or ref in already:
            # A document that changed should not leave its old claims behind,
            # and re-reading it without forgetting first would keep both.
            removed += graph.forget([ref])

        graph.add(result.entities, result.edges)
        extracted.append(result)

    return GraphReport(
        extracted=extracted, failed=failed, skipped=skipped, removed=removed
    )


def _ref_for(path, root) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
