"""Score SQLite retrieval against questions with known source documents.

Usage:
    python examples/library/evaluate_retrieval.py PROJECT QUESTIONS [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, evaluate, load_questions
from osintgpt.llm import build_embedding_provider
from osintgpt.vector_store import store_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure whether retrieval finds known answer documents.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('questions', type=Path, help='question-set TOML')
    parser.add_argument('--embedding-provider', help='provider id')
    parser.add_argument('--embedding-model', help='model name')
    parser.add_argument('--top-k', type=int, default=10, help='retrieval depth')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    if project.settings.storage_backend != 'sqlite':
        raise SystemExit('this example uses the default SQLite store')

    config = project.settings_for(Settings.from_env())
    provider = arguments.embedding_provider or project.settings.embedding_provider
    model = arguments.embedding_model or project.settings.embedding_model or None
    embedder = build_embedding_provider(provider, config, model=model)

    store = store_for(project, config)
    try:
        known_refs = store.refs(embedder.model)
    finally:
        store.close()

    report = evaluate(
        project,
        load_questions(arguments.questions),
        embedder,
        top_k=arguments.top_k,
        known_refs=known_refs,
    )
    print(report.summary)
    for result in report.misses:
        print(f'missed: {result.question.text}')
    for problem in report.unscorable:
        print(f'unscorable: {problem}')


if __name__ == '__main__':
    main()
