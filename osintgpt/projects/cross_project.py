# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: cross_project.py
# Description: Searching several projects at once. Vectors from different
#   embedding models are not comparable, so projects that do not share one are
#   dropped and said so rather than quietly ranked together.
# =================================================================================

# import submodules
from collections import Counter
from dataclasses import dataclass, field

# type hints
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

# import osintgpt config
from osintgpt.config import DEFAULT_EMBEDDING_MODEL

from .project import Project
from .settings import ProjectSettings

MISMATCH = 'different embedding model'

# One project's contribution to a cross-project search: (score, payload) pairs.
# Scores must come from the same embedding model to be comparable, which
# select_projects guarantees before this is ever called.
ProjectQuery = Callable[[Project], Iterable[Tuple[float, Any]]]


# Exclusion class
@dataclass(frozen=True)
class Exclusion:
    '''
    A project left out of a search, and why.
    '''
    slug: str
    reason: str
    detail: str = ''


# ProjectSelection class
@dataclass(frozen=True)
class ProjectSelection:
    '''
    Which projects a cross-project search may merge, and which it dropped.
    '''
    embedding_model: str
    included: List[Project] = field(default_factory=list)
    excluded: List[Exclusion] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.included) + len(self.excluded)

    # operator-facing summary
    @property
    def notice(self) -> str:
        '''
        What was skipped, phrased for a human. Empty when nothing was.

        Returns:
            str: A one-line summary naming the excluded projects.
        '''
        if not self.excluded:
            return ''

        named = ', '.join(
            f'{e.slug} ({e.detail})' if e.detail else e.slug
            for e in self.excluded
        )

        return (
            f'{len(self.excluded)} of {self.total} projects skipped — '
            f'{MISMATCH}: {named}'
        )


# CrossProjectHit class
@dataclass(frozen=True)
class CrossProjectHit:
    '''
    One result, carrying the project it came from so a citation can name it.
    '''
    project_slug: str
    score: float
    payload: Any


# CrossProjectResults class
@dataclass(frozen=True)
class CrossProjectResults:
    '''
    Merged results plus the selection that produced them. The selection travels
    with the answer so a caller cannot report the hits without the caveat.
    '''
    hits: List[CrossProjectHit] = field(default_factory=list)
    selection: Optional[ProjectSelection] = None

    @property
    def notice(self) -> str:
        return self.selection.notice if self.selection else ''

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)


# the embedding model a project actually searches with
def embedding_model_of(
    project: Project, defaults: Optional[ProjectSettings] = None
) -> str:
    '''
    Args:
        project (Project): Project to inspect.
        defaults (ProjectSettings, optional): User defaults filling what the \
            project left unset.

    Returns:
        str: The embedding model, falling back to the library default.
    '''
    settings = project.effective_settings(defaults)

    return settings.embedding_model or DEFAULT_EMBEDDING_MODEL


# decide which projects may be searched together
def select_projects(
    projects: Sequence[Project],
    embedding_model: Optional[str] = None,
    defaults: Optional[ProjectSettings] = None
) -> ProjectSelection:
    '''
    Split projects into those sharing an embedding model and those that do not.

    When no model is named, the most common one among the projects wins, ties
    broken by the order given — so listing projects that mostly agree keeps
    the majority rather than whichever happened to be first.

    Args:
        projects (Sequence[Project]): Projects the caller chose to search.
        embedding_model (str, optional): Force a target model instead of \
            inferring one.
        defaults (ProjectSettings, optional): User defaults.

    Returns:
        ProjectSelection: Included projects, exclusions, and the target model.
    '''
    unique: List[Project] = []
    seen = set()
    for project in projects:
        if project.id in seen:
            continue
        seen.add(project.id)
        unique.append(project)

    if not unique:
        return ProjectSelection(embedding_model=embedding_model or '')

    models = {p.id: embedding_model_of(p, defaults) for p in unique}
    if embedding_model is None:
        counts = Counter(models[p.id] for p in unique)
        best = max(counts.values())
        embedding_model = next(
            models[p.id] for p in unique if counts[models[p.id]] == best
        )

    included, excluded = [], []
    for project in unique:
        if models[project.id] == embedding_model:
            included.append(project)
        else:
            excluded.append(Exclusion(
                slug=project.slug, reason=MISMATCH, detail=models[project.id]
            ))

    return ProjectSelection(
        embedding_model=embedding_model, included=included, excluded=excluded
    )


# run one search across several projects
def search_projects(
    projects: Sequence[Project],
    query: ProjectQuery,
    embedding_model: Optional[str] = None,
    defaults: Optional[ProjectSettings] = None,
    limit: Optional[int] = None
) -> CrossProjectResults:
    '''
    Search every compatible project and merge the results by score.

    Merging is only sound because the selection guarantees one embedding model
    across the included projects; `query` is never called for an excluded one.

    Args:
        projects (Sequence[Project]): Projects the caller chose to search.
        query (ProjectQuery): Runs the search for one project and yields \
            (score, payload) pairs.
        embedding_model (str, optional): Force a target model.
        defaults (ProjectSettings, optional): User defaults.
        limit (int, optional): Keep only the top N merged hits.

    Returns:
        CrossProjectResults: Merged hits and the selection behind them.
    '''
    selection = select_projects(projects, embedding_model, defaults)

    hits: List[CrossProjectHit] = []
    for project in selection.included:
        hits.extend(
            CrossProjectHit(
                project_slug=project.slug, score=float(score), payload=payload
            )
            for score, payload in query(project)
        )

    hits.sort(key=lambda hit: hit.score, reverse=True)
    if limit is not None:
        hits = hits[:limit]

    return CrossProjectResults(hits=hits, selection=selection)
