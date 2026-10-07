# -*- coding: utf-8 -*-

"""
Batch ask — answer a list of questions from one project and keep every answer as JSONL.

Each line of the output holds the question, answer, sources, follow-ups, the
full tool trace, and that answer's token usage. Lines are written as answers
arrive, so an interrupted run keeps what it finished; --resume skips questions
already in the output. A cost ceiling covers the whole batch.

Usage:

    python batch_ask.py <project-dir> <questions> --out answers.jsonl [options]

    python batch_ask.py ~/.osintgpt/projects/my-case questions.txt --out answers.jsonl
    python batch_ask.py ~/.osintgpt/projects/my-case questions.toml \\
                        --out answers.jsonl --ceiling 0.50 --resume

Questions come from a text file (one per line, # for comments) or an
evaluation set (.toml), whose text fields are asked and expected refs ignored.
"""

# import modules
import argparse
import json
import sys

# import submodules
from pathlib import Path

# import osintgpt modules
from osintgpt import Project, Settings, agentic_answer, load_questions
from osintgpt.agentic import answer_to_dict
from osintgpt.llm import (
    UsageRecorder,
    build_embedding_provider,
    build_generation_provider
)
from osintgpt.llm.usage import CostLimitReached
from osintgpt.vector_store import store_for


def read_questions(path):
    """Questions in file order, from .toml evaluation sets or plain text."""
    path = Path(path)
    if path.suffix.lower() == '.toml':
        return [question.text for question in load_questions(path)]

    lines = path.read_text(encoding='utf-8').splitlines()

    return [line.strip() for line in lines if line.strip() and not line.startswith('#')]


def answered(path):
    """Questions already present in an earlier output file."""
    if not Path(path).is_file():
        return set()

    with open(path, encoding='utf-8') as stream:
        return {json.loads(line)['question'] for line in stream if line.strip()}


def main():
    parser = argparse.ArgumentParser(
        description='Answer many questions and write each answer as one JSON line.'
    )
    parser.add_argument('project', help='project directory')
    parser.add_argument('questions', help='.txt (one per line) or .toml question set')
    parser.add_argument('--out', required=True, help='JSONL file to write')
    parser.add_argument('--ceiling', type=float, help='stop the batch above this USD estimate')
    parser.add_argument('--resume', action='store_true', help='skip questions already in --out')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    questions = read_questions(arguments.questions)
    done = answered(arguments.out) if arguments.resume else set()
    pending = [question for question in questions if question not in done]
    if not pending:
        print('nothing to ask')
        return 0

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env(arguments.env_file))
    recorder = UsageRecorder(cost_ceiling_usd=arguments.ceiling)
    embedder = build_embedding_provider(
        project.settings.embedding_provider, config,
        model=project.settings.embedding_model or None, recorder=recorder
    )
    generator = build_generation_provider(
        project.settings.generation_provider, config,
        model=project.settings.generation_model or None, recorder=recorder
    )

    mode = 'a' if arguments.resume else 'w'
    store = store_for(project, config)
    status = 0
    try:
        with open(arguments.out, mode, encoding='utf-8') as stream:
            for position, question in enumerate(pending, 1):
                before = (recorder.calls, recorder.total_tokens)
                answer = agentic_answer(project, question, embedder, generator, store=store)

                record = answer_to_dict(answer)
                record['usage'] = {
                    'calls': recorder.calls - before[0],
                    'tokens': recorder.total_tokens - before[1]
                }
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                print(f'{position}/{len(pending)} {answer.trace.summary}: {question}')
    except CostLimitReached as error:
        # Finished answers are already on disk; --resume continues from here.
        print(f'\n{error}')
        status = 1
    finally:
        if hasattr(store, 'close'):
            store.close()

    print(f'\nUsage: {recorder.summary}')

    return status


if __name__ == '__main__':
    sys.exit(main())
