"""Ask follow-up questions in a recorded thread, carrying recent turns as context.

Usage:
    python examples/library/conversation_thread.py PROJECT QUESTION [QUESTION ...]
    python examples/library/conversation_thread.py PROJECT QUESTION --continue ID
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, agentic_answer
from osintgpt.agentic import answer_to_dict
from osintgpt.llm import UsageRecorder, build_embedding_provider, build_generation_provider
from osintgpt.projects import (
    append_turn,
    open_conversation,
    recent_pairs,
    start_conversation,
)
from osintgpt.vector_store import store_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Ask questions in one conversation thread, as `ask --conversation` does.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('questions', nargs='+', help='asked in order, in one thread')
    parser.add_argument('--continue', dest='thread_id', metavar='ID',
                        help='conversation id to continue')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    if arguments.thread_id:
        thread = open_conversation(project, arguments.thread_id)
        if thread is None:
            raise SystemExit(f'no conversation {arguments.thread_id!r} in {project.slug}')
    else:
        thread = start_conversation(project, arguments.questions[0])

    config = project.settings_for(Settings.from_env(arguments.env_file))
    recorder = UsageRecorder()
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
        for question in arguments.questions:
            # Read the window before recording, so it never holds this question.
            carried = recent_pairs(thread, project.settings.conversation_window)
            answer = agentic_answer(
                project, question, embedder, generator,
                store=store, conversation=carried,
            )
            append_turn(thread, question, answer_to_dict(answer))

            print(f'Q: {question}')
            print(f'   carried {len(carried)} earlier turn(s); trace: {answer.trace.summary}')
            print(f'A: {answer.text}\n')
    finally:
        if hasattr(store, 'close'):
            store.close()

    print(f'Conversation {thread.id} ({len(thread)} turns) in {thread.path.parent}')
    print(f'Usage: {recorder.summary}')


if __name__ == '__main__':
    main()
