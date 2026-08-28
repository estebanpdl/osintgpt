# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: answering.py
# Description: Answering a question from a project's own documents. Retrieval
#   decides what the model may use; the answer travels with the passages it
#   was built from.
# =================================================================================

# import submodules
from dataclasses import dataclass, field

# type hints
from typing import Iterable, List, Optional, Sequence

# import osintgpt llm
from osintgpt.llm.base import EmbeddingProvider, GenerationProvider

# import osintgpt projects
from osintgpt.projects import Project

# import osintgpt prompts
from osintgpt.prompts import prompt

# import osintgpt search
from osintgpt.search import search_project

# import osintgpt vector store
from osintgpt.vector_store import BaseVectorEngine, SearchResult

# Enough passages to answer from several documents, few enough that the model
# is not asked to weigh a corpus. The number an analyst would read.
DEFAULT_PASSAGES = 8

# Said when retrieval found nothing, without spending a generation call. A
# model asked to answer from no passages will answer from its training, which
# is the failure this whole design exists to prevent.
NOTHING_RETRIEVED = (
    'Nothing in this project matches that question closely enough to answer '
    'from. The corpus may not cover it, or it may not be indexed yet.'
)


# Answer class
@dataclass(frozen=True)
class Answer:
    '''
    A grounded answer and the passages it was allowed to use.
    '''
    question: str
    text: str
    # In the order the model saw them, so a citation of [2] resolves here.
    passages: List[SearchResult] = field(default_factory=list)
    # False when retrieval found nothing and no model was called.
    generated: bool = False

    @property
    def sources(self) -> List[str]:
        '''
        Returns:
            List[str]: Documents behind the answer, deduplicated, in the \
                order the passages ranked.
        '''
        seen: List[str] = []
        for result in self.passages:
            if result.chunk.citation not in seen:
                seen.append(result.chunk.citation)

        return seen

    @property
    def citations(self) -> List[str]:
        '''
        Returns:
            List[str]: One line per passage, numbered as the model saw them, \
                so a reader can follow [2] back to a document.
        '''
        return [
            f'[{index}] {result.chunk.citation}'
            for index, result in enumerate(self.passages, 1)
        ]


# answer a question from a project's documents
def answer_question(
    project: Project,
    question: str,
    embedder: EmbeddingProvider,
    generator: GenerationProvider,
    passages: int = DEFAULT_PASSAGES,
    refs: Optional[Iterable[str]] = None,
    store: Optional[BaseVectorEngine] = None
) -> Answer:
    '''
    Retrieve, then answer from what was retrieved and nothing else.

    Retrieval runs first and decides what the model may use, which is what
    makes the answer checkable: every claim should trace to a passage a reader
    can open.

    Args:
        project (Project): The indexed project to answer from.
        question (str): The question, as asked.
        embedder (EmbeddingProvider): Must be the model the project was \
            indexed with.
        generator (GenerationProvider): Writes the answer.
        passages (int): How many passages to retrieve and offer.
        refs (Iterable[str], optional): Restrict to these documents.
        store (BaseVectorEngine, optional): Defaults to the project's own.

    Returns:
        Answer: The answer and the passages behind it.
    '''
    found = search_project(
        project, question, embedder, top_k=passages, refs=refs, store=store
    )

    if not found:
        # No call is made. A model given no passages answers from its
        # training, which is exactly what grounding is for.
        return Answer(question=question, text=NOTHING_RETRIEVED)

    return Answer(
        question=question,
        text=generator.generate(
            build_prompt(question, found), question
        ).strip(),
        passages=list(found),
        generated=True
    )


# assemble the prompt a question is answered from
def build_prompt(question: str, passages: Sequence[SearchResult]) -> str:
    '''
    Args:
        question (str): The question, as asked.
        passages (Sequence[SearchResult]): Retrieved passages, best first.

    Returns:
        str: The system prompt, with the passages numbered from one so a \
            citation of [2] means the second passage.
    '''
    return prompt(
        'answer',
        question=question,
        passages=[
            {'citation': result.chunk.citation, 'text': result.text}
            for result in passages
        ]
    )
