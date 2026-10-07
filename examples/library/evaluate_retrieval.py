"""Score retrieval methods against questions with known source documents.

Usage:
    python examples/library/evaluate_retrieval.py PROJECT QUESTIONS [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, evaluate, load_questions
from osintgpt.evaluation import RETRIEVAL_METHODS
from osintgpt.llm import UsageRecorder, build_embedding_provider
from osintgpt.vector_store import store_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure whether retrieval finds known answer documents.',
        epilog='Hybrid evaluation reads terms = [...] from each question.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('questions', type=Path, help='question-set TOML')
    parser.add_argument('--top-k', type=int, default=10, help='retrieval depth')
    parser.add_argument(
        '--retrieval', choices=RETRIEVAL_METHODS, action='append',
        help='method to measure; repeatable (default: every method)'
    )
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env(arguments.env_file))
    recorder = UsageRecorder()
    embedder = build_embedding_provider(
        project.settings.embedding_provider,
        config,
        model=project.settings.embedding_model or None,
        recorder=recorder,
    )
    questions = load_questions(arguments.questions)

    store = store_for(project, config)
    try:
        # Expected refs absent from the store are reported, not scored as misses.
        known_refs = store.refs(embedder.model)
        for method in arguments.retrieval or RETRIEVAL_METHODS:
            report = evaluate(
                project,
                questions,
                embedder,
                top_k=arguments.top_k,
                known_refs=known_refs,
                retrieval=method,
                store=store,
            )
            print(f'{report.retrieval}: {report.summary}')
            for result in report.misses:
                print(f'  missed: {result.question.text}')
            for problem in report.unscorable:
                print(f'  unscorable: {problem}')
    finally:
        if hasattr(store, 'close'):
            store.close()

    print(f'Usage: {recorder.summary}')


if __name__ == '__main__':
    main()
