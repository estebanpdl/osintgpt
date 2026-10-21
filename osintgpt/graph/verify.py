'''Check that graph evidence is present in its claimed source document.'''

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot, read_snapshot
from osintgpt.ingestion.provenance import quote_locations
from osintgpt.ingestion.source_view import source_state
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
    snapshot_status: str = 'unavailable'
    source_state: str = 'unversioned'
    current_version: str = ''
    snapshot_problem: str = ''
    current_locations: list = field(default_factory=list)

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


def _path_for(ref: str, root: Path) -> Path:
    path = Path(ref)

    return path if path.is_absolute() else root / path


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

    return verify_claims(project, edges)


def verify_claims(project, edges, *, first_only=False, require_registered=False, cache=None):
    from .reconciliation import registered

    if cache is not None:
        if cache.project.id != project.id or cache.project.paths.root != project.paths.root:
            raise ValueError('Evidence cache belongs to a different project.')
        cache.bind_policy(require_registered)
        cache.check()
    corpus = cache.corpus if cache is not None else Corpus.load(project.paths.sources)
    root = Path(project.paths.root)
    documents, saved, states = (cache.documents, cache.saved, cache.states) if cache is not None else ({}, {}, {})
    results = []
    for edge in edges:
        result_key = (edge.assertion_id, first_only, require_registered)
        if cache is not None and result_key in cache.results:
            results.append(cache.results[result_key])
            continue
        ref = edge.provenance.get('source_ref', edge.ref)
        path = _path_for(ref, root)
        mapping = cache.mapping_for(path) if cache is not None else corpus.mapping_for(path, root)
        if ref not in documents:
            if cache is not None:
                cache.watch_source(ref, path, mapping, not require_registered or registered(corpus, root, ref))
            try:
                if require_registered and not registered(corpus, root, ref):
                    raise ValueError('Source is no longer registered; current text was not read.')
                current = load_snapshot(path, mapping, ref=ref, cache_dir=project.paths.extracts,
                                        **({'transcriber': cache.read_transcription} if cache is not None else {}))
                documents[ref] = (current, '')
            except Exception as error:
                documents[ref] = (None, 'document does not exist'
                                  if isinstance(error, FileNotFoundError) else str(error))
        current, problem = documents[ref]
        if cache is not None:
            cache.watch_transcriptions(current)
        document = current.document(edge.ref) if current else None
        quotes = edge.evidence_quotes or (edge.evidence,)
        matched_quotes = [quote_locations(document, quote, first_only=first_only) if document else [] for quote in quotes]
        matches = [match for locations in matched_quotes for match in locations] if all(matched_quotes) else []
        status = UNREADABLE if problem else FOUND if matches else NOT_FOUND
        snapshot_status, state, snapshot_problem = 'unavailable', 'unversioned', ''
        version = edge.provenance.get('source_version', '')
        if version:
            state = 'unknown'
            if version not in saved:
                if cache is not None:
                    cache.watch_saved(version)
                try:
                    saved[version] = (current if current and current.version == version
                                      else read_snapshot(project.paths.extracts, version), '')
                except Exception as error:
                    saved[version] = (None, str(error))
            snapshot, snapshot_problem = saved[version]
            if cache is not None:
                cache.watch_transcriptions(snapshot)
            if snapshot and snapshot.ref != ref:
                snapshot, snapshot_problem = None, 'Saved text belongs to a different source.'
            if snapshot:
                original = snapshot.document(edge.ref)
                snapshot_status = FOUND if original and all(quote_locations(original, quote, first_only=True) for quote in quotes) else NOT_FOUND
                key = (ref, version)
                if key not in states:
                    states[key] = (source_state(path, mapping, snapshot, project.paths.extracts)
                                   if not require_registered or registered(corpus, root, ref) else 'unregistered')
                state = states[key]
        results.append(EvidenceResult(
            edge=edge, status=status, problem=problem, snapshot_status=snapshot_status,
            source_state=state, snapshot_problem=snapshot_problem,
            current_version=current.version if current else '', current_locations=matches
        ))
        if cache is not None:
            cache.results[result_key] = results[-1]

    if cache is not None:
        cache.check()
    return EvidenceReport(results=results)
