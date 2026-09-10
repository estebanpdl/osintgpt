"""Run the one-pass and model-directed answer paths for one question.

Usage:
    python examples/library/answer_with_citations.py PROJECT QUESTION [options]
"""

import argparse
from pathlib import Path

from osintgpt import (
    Project,
    Settings,
    agentic_answer,
    answer_question,
)
from osintgpt.llm import build_embedding_provider, build_generation_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print static and agentic answers with their provenance.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('question', help='question to answer')
    parser.add_argument('--embedding-provider', help='provider id')
    parser.add_argument('--embedding-model', help='model name')
    parser.add_argument('--generation-provider', help='provider id')
    parser.add_argument('--generation-model', help='model name')
    parser.add_argument('--passages', type=int, default=8, help='static limit')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env())
    embedding_provider = (
        arguments.embedding_provider or project.settings.embedding_provider
    )
    generation_provider = (
        arguments.generation_provider or project.settings.generation_provider
    )
    embedder = build_embedding_provider(
        embedding_provider,
        config,
        model=arguments.embedding_model or project.settings.embedding_model or None,
    )
    generator = build_generation_provider(
        generation_provider,
        config,
        model=arguments.generation_model or project.settings.generation_model or None,
    )

    static = answer_question(
        project,
        arguments.question,
        embedder,
        generator,
        passages=arguments.passages,
    )
    print('Static answer')
    print(static.text)
    for citation in static.citations:
        print(citation)

    agentic = agentic_answer(project, arguments.question, embedder, generator)
    print('\nAgentic answer')
    print(agentic.text)
    for source in agentic.sources:
        print(source)
    print('\nTrace')
    for line in agentic.trace.lines() + agentic.trace.reading:
        print(line)


if __name__ == '__main__':
    main()
