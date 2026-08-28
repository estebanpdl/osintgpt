'''Check that graph evidence is present in its claimed source document.'''

import unicodedata

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from osintgpt.ingestion import Corpus, load_documents
from osintgpt.projects import Project

from .store import Edge, graph_for

FOUND = 'found'
NOT_FOUND = 'not_found'
UNREADABLE = 'unreadable'


@dataclass(frozen=True)
class EvidenceResult:
    '''The verification outcome for one sourced graph edge.'''
    edge: Edge
    status: str
    problem: str = ''

    @property
    def ok(self) -> bool:
        return self.status == FOUND


@dataclass(frozen=True)
class EvidenceReport:
    '''Evidence outcomes and their aggregate counts.'''
    results: List[EvidenceResult] = field(default_factory=list)

    @property
    def found(self) -> int:
        return sum(result.status == FOUND for result in self.results)

    @property
    def not_found(self) -> int:
        return sum(result.status == NOT_FOUND for result in self.results)

    @property
    def unreadable(self) -> int:
        return sum(result.status == UNREADABLE for result in self.results)

    @property
    def failures(self) -> List[EvidenceResult]:
        return [result for result in self.results if not result.ok]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def summary(self) -> str:
        parts = []
        if self.found:
            parts.append(f'{self.found} verified')
        if self.not_found:
            parts.append(f'{self.not_found} not found')
        if self.unreadable:
            parts.append(f'{self.unreadable} unreadable')

        return ', '.join(parts) if parts else 'nothing to verify'


@dataclass(frozen=True)
class _DocumentText:
    text: str = ''
    problem: str = ''


def _normalized(text: str) -> str:
    return ' '.join(unicodedata.normalize('NFC', text).split())


def _path_for(ref: str, root: Path) -> Path:
    path = Path(ref)

    return path if path.is_absolute() else root / path


def _read_document(path: Path, corpus: Corpus, root: Path) -> _DocumentText:
    if not path.is_file():
        return _DocumentText(problem='document does not exist')

    try:
        documents = load_documents(path, corpus.mapping_for(path, root))
        text = '\n\n'.join(
            document.text for document in documents if document.text
        ).strip()
    except Exception as error:  # noqa: BLE001 — one source, not the report
        return _DocumentText(problem=str(error))

    return _DocumentText(text=_normalized(text))


def verify_evidence(
    project: Project,
    refs: Optional[Iterable[str]] = None
) -> EvidenceReport:
    '''
    Check every selected edge quote against its claimed source document.

    Args:
        project (Project): The project whose graph and corpus to check.
        refs (Iterable[str], optional): Restrict checks to these documents.

    Returns:
        EvidenceReport: One non-mutating result per selected edge.
    '''
    graph_path = Path(project.paths.root) / 'graph.sqlite'
    if not graph_path.is_file():
        return EvidenceReport()

    with graph_for(project) as graph:
        edges = graph.edges(refs=refs)

    corpus = Corpus.load(project.paths.sources)
    root = Path(project.paths.root)
    documents: Dict[str, _DocumentText] = {}
    results = []
    for edge in edges:
        if edge.ref not in documents:
            documents[edge.ref] = _read_document(
                _path_for(edge.ref, root), corpus, root
            )

        document = documents[edge.ref]
        evidence = _normalized(edge.evidence)
        if document.problem:
            status = UNREADABLE
        elif evidence and evidence in document.text:
            status = FOUND
        else:
            status = NOT_FOUND
        results.append(EvidenceResult(
            edge=edge, status=status, problem=document.problem
        ))

    return EvidenceReport(results=results)
