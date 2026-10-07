"""Answer one question through the one-pass and model-directed paths, with provenance.

Usage:
    python examples/library/answer_with_citations.py PROJECT QUESTION [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, agentic_answer, answer_question
from osintgpt.llm import UsageRecorder, build_embedding_provider, build_generation_provider
from osintgpt.llm.usage import CostLimitReached
from osintgpt.vector_store import store_for


def announce(round_number: int, turn) -> None:
    """Print each tool request as the model makes it."""
    for call in turn.calls:
        print(f'  round {round_number}: {call.name}({call.arguments})')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print static and agentic answers with their sources and trace.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('question', help='question to answer')
    parser.add_argument('--mode', choices=('static', 'agentic', 'both'), default='both')
    parser.add_argument('--passages', type=int, default=8, help='static passage limit')
    parser.add_argument('--ceiling', type=float, help='stop above this USD estimate')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env(arguments.env_file))
    recorder = UsageRecorder(cost_ceiling_usd=arguments.ceiling)
    embedder = build_embedding_provider(
        project.settings.embedding_provider,
        config,
        model=project.settings.embedding_model or None,
        recorder=recorder,
    )
    generator = build_generation_provider(
        project.settings.generation_provider,
        config,
        model=project.settings.generation_model or None,
        recorder=recorder,
    )

    store = store_for(project, config)
    try:
        if arguments.mode in ('static', 'both'):
            static = answer_question(
                project, arguments.question, embedder, generator,
                passages=arguments.passages, store=store,
            )
            print('Static answer (one semantic retrieval, then one answer)')
            print(static.text)
            print('\nPassages the model saw')
            for citation in static.citations:
                print(f'  {citation}')

        if arguments.mode in ('agentic', 'both'):
            print('\nAgentic answer (the model chooses its tools)')
            agentic = agentic_answer(
                project, arguments.question, embedder, generator,
                store=store, on_round=announce,
            )
            print(agentic.text)
            print('\nSources')
            for source in agentic.sources:
                print(f'  {source}')
            print(f'\nTrace: {agentic.trace.summary}')
            for line in agentic.trace.lines() + agentic.trace.reading:
                print(f'  {line}')
            if agentic.followups:
                print('\nAsk next')
                for number, followup in enumerate(agentic.followups, 1):
                    print(f'  {number}. {followup}')
    except CostLimitReached as error:
        raise SystemExit(f'{error}\nUsage: {recorder.summary}')
    finally:
        if hasattr(store, 'close'):
            store.close()

    print(f'\nUsage: {recorder.summary}')


if __name__ == '__main__':
    main()
