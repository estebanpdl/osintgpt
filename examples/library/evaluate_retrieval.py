"""Score retrieval against questions with known source documents.

Usage:
    python examples/library/evaluate_retrieval.py PROJECT QUESTIONS [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, evaluate, load_questions
from osintgpt.evaluation import RETRIEVAL_METHODS
from osintgpt.llm import build_embedding_provider
from osintgpt.vector_store import store_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure whether retrieval finds known answer documents.',
        epilog='Hybrid evaluation reads terms = [...] from each question.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('questions', type=Path, help='question-set TOML')
    parser.add_argument('--embedding-provider', help='provider id')
    parser.add_argument('--embedding-model', help='model name')
    parser.add_argument('--top-k', type=int, default=10, help='retrieval depth')
    parser.add_argument(
        '--retrieval', choices=RETRIEVAL_METHODS,
        default=RETRIEVAL_METHODS[0],
        help='retrieval method to measure'
    )
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env())
    provider = (
        arguments.embedding_provider or project.settings.embedding_provider
    )
    model = (
        arguments.embedding_model or project.settings.embedding_model or None
    )
    embedder = build_embedding_provider(provider, config, model=model)

    store = store_for(project, config)
    try:
        known_refs = store.refs(embedder.model)
        report = evaluate(
            project,
            load_questions(arguments.questions),
            embedder,
            top_k=arguments.top_k,
            known_refs=known_refs,
            retrieval=arguments.retrieval,
            store=store,
        )
    finally:
        store.close()

    print(f'{report.retrieval}: {report.summary}')
    for result in report.misses:
        print(f'missed: {result.question.text}')
    for problem in report.unscorable:
        print(f'unscorable: {problem}')


if __name__ == '__main__':
    main()
