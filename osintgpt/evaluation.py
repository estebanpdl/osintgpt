# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: evaluation.py
# Description: Scoring retrieval against questions whose answers are known.
#   The difference between arguing that a change helped and knowing it did.
# =================================================================================

# import modules
import logging

# import submodules
from dataclasses import dataclass, field
from pathlib import Path

# type hints
from typing import Dict, List, Optional, Sequence, Union

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider

# import osintgpt projects
from osintgpt.projects import Project
from osintgpt.projects.toml_io import read_toml, write_toml

# import osintgpt search
from osintgpt.search import hybrid_search, search_project

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, store_for

log = logging.getLogger('osintgpt.evaluation')

# How deep to look when asking whether retrieval found the right document.
# Ten is roughly what a reading model would be given, so a document below it
# is one an answer would not have seen.
DEFAULT_TOP_K = 10

SEMANTIC = 'semantic'
HYBRID = 'hybrid'
RETRIEVAL_METHODS = (SEMANTIC, HYBRID)

QUESTIONS_HEADER = '''\
# osintgpt evaluation set
#
# Questions whose answers are known, so a retrieval change can be measured
# rather than argued about. Keep them real: questions an analyst would
# actually ask of this corpus, with the documents that genuinely answer them.
# A set written to make the numbers look good measures nothing.
# Optional terms are literal strings used only by hybrid evaluation.

'''


# Question class
@dataclass(frozen=True)
class Question:
    '''
    One question and the documents that should answer it.
    '''
    text: str
    # Refs as the store holds them. A question with none is a question nobody
    # can score, so it is reported rather than counted.
    expected: List[str] = field(default_factory=list)
    # Free text: why this question is in the set, or what it is probing.
    note: str = ''
    # Literal strings for the lexical leg of a hybrid evaluation.
    terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        recorded = {'text': self.text, 'expected': list(self.expected)}
        if self.terms:
            recorded['terms'] = list(self.terms)
        if self.note:
            recorded['note'] = self.note

        return recorded

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=str(data.get('text', '')),
            expected=[str(ref) for ref in data.get('expected', [])],
            terms=[str(term) for term in data.get('terms', [])],
            note=str(data.get('note', ''))
        )


# QuestionResult class
@dataclass(frozen=True)
class QuestionResult:
    '''
    How retrieval did on one question.
    '''
    question: Question
    # Refs returned, best first.
    retrieved: List[str] = field(default_factory=list)
    # Position of the first expected document, 1-based. None when none of
    # them came back at all.
    first_hit: Optional[int] = None

    @property
    def found(self) -> bool:
        return self.first_hit is not None

    @property
    def recall(self) -> float:
        '''
        Returns:
            float: Share of the expected documents that came back. Recall \
                rather than precision: a passage the analyst never sees is \
                the failure that matters, and extra results cost attention \
                rather than answers.
        '''
        if not self.question.expected:
            return 0.0

        found = sum(
            1 for ref in self.question.expected if ref in self.retrieved
        )

        return found / len(self.question.expected)

    @property
    def reciprocal_rank(self) -> float:
        '''
        Returns:
            float: 1/rank of the first expected document, 0.0 when absent. \
                Rewards finding the right thing early, which is what a \
                reading model with a limited window actually needs.
        '''
        return 1.0 / self.first_hit if self.first_hit else 0.0


# EvaluationReport class
@dataclass(frozen=True)
class EvaluationReport:
    '''
    How retrieval did across a question set.
    '''
    results: List[QuestionResult] = field(default_factory=list)
    # Questions that could not be scored: no expected documents, or expected
    # documents the store has never heard of.
    unscorable: List[str] = field(default_factory=list)
    embedding_model: str = ''
    top_k: int = DEFAULT_TOP_K
    retrieval: str = SEMANTIC

    @property
    def scored(self) -> int:
        return len(self.results)

    @property
    def found(self) -> int:
        return sum(1 for result in self.results if result.found)

    @property
    def hit_rate(self) -> float:
        '''
        Returns:
            float: Share of questions where an expected document appeared at \
                all. The blunt measure, and the one that catches a change \
                making retrieval worse.
        '''
        return self.found / self.scored if self.scored else 0.0

    @property
    def mean_recall(self) -> float:
        if not self.results:
            return 0.0

        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def mean_reciprocal_rank(self) -> float:
        '''
        Returns:
            float: Mean 1/rank of the first expected document. More sensitive \
                than the hit rate: a change that moves the right document \
                from position eight to position two shows here and nowhere \
                else.
        '''
        if not self.results:
            return 0.0

        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def misses(self) -> List[QuestionResult]:
        '''
        Returns:
            List[QuestionResult]: Questions where nothing expected came back. \
                The list worth reading — an aggregate says a change helped, \
                these say what it still cannot answer.
        '''
        return [result for result in self.results if not result.found]

    @property
    def summary(self) -> str:
        if not self.results:
            return 'nothing scored'

        parts = [
            f'{self.found}/{self.scored} found',
            f'hit rate {self.hit_rate:.0%}',
            f'MRR {self.mean_reciprocal_rank:.3f}',
            f'recall {self.mean_recall:.0%}'
        ]
        if self.unscorable:
            parts.append(f'{len(self.unscorable)} unscorable')

        return ', '.join(parts)

    # compare against an earlier run
    def against(self, previous: 'EvaluationReport') -> str:
        '''
        Args:
            previous (EvaluationReport): An earlier run to compare with.

        Returns:
            str: What moved, phrased so a regression is as visible as a gain.
        '''
        deltas = [
            ('hit rate', self.hit_rate - previous.hit_rate, '{:+.0%}'),
            (
                'MRR',
                self.mean_reciprocal_rank - previous.mean_reciprocal_rank,
                '{:+.3f}'
            ),
            ('recall', self.mean_recall - previous.mean_recall, '{:+.0%}')
        ]
        moved = [
            f'{name} {template.format(delta)}'
            for name, delta, template in deltas
            if abs(delta) > 1e-9
        ]

        return ', '.join(moved) if moved else 'no change'


# read a question set
def load_questions(path: Union[str, Path]) -> List[Question]:
    '''
    Args:
        path (Union[str, Path]): The question set file.

    Returns:
        List[Question]: The questions, empty when the file is absent.
    '''
    document = read_toml(path)

    return [
        Question.from_dict(row) for row in document.get('question', [])
    ]


# write a question set
def save_questions(
    path: Union[str, Path], questions: Sequence[Question]
) -> None:
    '''
    Args:
        path (Union[str, Path]): Where to write.
        questions (Sequence[Question]): The set.
    '''
    write_toml(
        path,
        {'question': [question.to_dict() for question in questions]},
        header=QUESTIONS_HEADER
    )


# score retrieval against a question set
def evaluate(
    project: Project,
    questions: Sequence[Question],
    embedder: EmbeddingProvider,
    top_k: int = DEFAULT_TOP_K,
    known_refs: Optional[Sequence[str]] = None,
    retrieval: str = SEMANTIC,
    store: Optional[BaseVectorEngine] = None
) -> EvaluationReport:
    '''
    Ask every question and check whether the right documents came back.

    Args:
        project (Project): The indexed project to search.
        questions (Sequence[Question]): Questions with known answers.
        embedder (EmbeddingProvider): Must be the model the project was \
            indexed with.
        top_k (int): How deep to look. A document below this is one an answer \
            would not have seen.
        known_refs (Sequence[str], optional): Documents the store holds. When \
            given, a question expecting something absent is reported as \
            unscorable rather than counted as a miss — a typo in the set \
            should not read as a retrieval failure.
        retrieval (str): `semantic` for the existing vector-only measure, or \
            `hybrid` to fuse semantic results with each question's exact \
            terms. No terms are derived or generated during evaluation.
        store (BaseVectorEngine, optional): An open store to reuse for the \
            whole run. It remains owned by the caller; otherwise this \
            function opens the project's store once and closes it.

    Returns:
        EvaluationReport: Per-question results and the aggregates over them.
    '''
    if retrieval not in RETRIEVAL_METHODS:
        raise ValueError(
            f'unknown retrieval {retrieval!r}; use '
            f'{" or ".join(RETRIEVAL_METHODS)}'
        )

    owned = store is None
    engine = store if store is not None else store_for(project)
    try:
        return _evaluate(
            project, questions, embedder, engine, top_k,
            known_refs, retrieval
        )
    finally:
        if owned:
            engine.close()


def _evaluate(
    project: Project,
    questions: Sequence[Question],
    embedder: EmbeddingProvider,
    store: BaseVectorEngine,
    top_k: int,
    known_refs: Optional[Sequence[str]],
    retrieval: str
) -> EvaluationReport:
    known = set(known_refs) if known_refs is not None else None
    results: List[QuestionResult] = []
    unscorable: List[str] = []

    for question in questions:
        if not question.expected:
            unscorable.append(f'{question.text} — no expected documents')
            continue

        if known is not None:
            missing = [ref for ref in question.expected if ref not in known]
            if missing:
                unscorable.append(
                    f'{question.text} — not in the store: {", ".join(missing)}'
                )
                continue

        if retrieval == HYBRID:
            hits = hybrid_search(
                project, question.text, embedder, top_k=top_k,
                terms=question.terms, store=store
            )
        else:
            hits = search_project(
                project, question.text, embedder, top_k=top_k, store=store
            )

        # A document can produce several chunks; what is being scored is
        # whether the document was reached, so rank by its best chunk.
        retrieved: List[str] = []
        for hit in hits:
            if hit.ref not in retrieved:
                retrieved.append(hit.ref)

        first_hit = None
        for position, ref in enumerate(retrieved, 1):
            if ref in question.expected:
                first_hit = position
                break

        results.append(QuestionResult(
            question=question, retrieved=retrieved, first_hit=first_hit
        ))

    return EvaluationReport(
        results=results,
        unscorable=unscorable,
        embedding_model=embedder.model,
        top_k=top_k,
        retrieval=retrieval
    )
