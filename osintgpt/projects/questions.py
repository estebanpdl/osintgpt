# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: questions.py
# Description: What has been asked of a project. Append-only, local, and read
#   by anything that should not repeat itself.
# =================================================================================

# import modules
import json
import logging

# import submodules
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# type hints
from typing import List, Optional, Union

log = logging.getLogger('osintgpt.projects')

QUESTIONS_FILE = 'questions.jsonl'

# Enough to keep a suggestion from repeating a recent question without handing
# a model the project's whole history.
RECENT = 20


# AskedQuestion class
@dataclass(frozen=True)
class AskedQuestion:
    '''
    One question, as it was asked.
    '''
    text: str
    asked_at: str = ''

    def to_line(self) -> str:
        return json.dumps(
            {'text': self.text, 'asked_at': self.asked_at},
            ensure_ascii=False
        )


# where a project keeps its questions
def questions_file(project) -> Path:
    '''
    Args:
        project (Project): The project.

    Returns:
        Path: Its question log.
    '''
    return Path(project.paths.root) / QUESTIONS_FILE


# record a question
def record_question(project, text: str) -> Optional[AskedQuestion]:
    '''
    Append one question to the project's log.

    Append-only and one line per question, so a partial write costs the last
    line rather than the file, and a reader never has to parse a structure
    that something else is mid-way through rewriting.

    Failing to record is not failing to answer: a read-only directory or a
    full disk should cost the history, not the reply the analyst asked for.

    Args:
        project (Project): The project asked.
        text (str): The question, as asked.

    Returns:
        Optional[AskedQuestion]: What was recorded, or None if it could not be.
    '''
    cleaned = (text or '').strip()
    if not cleaned:
        return None

    asked = AskedQuestion(
        text=cleaned,
        asked_at=datetime.now(timezone.utc).isoformat(timespec='seconds')
    )

    try:
        path = questions_file(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(asked.to_line() + '\n')
    except OSError as error:
        log.warning('could not record the question: %s', error)

        return None

    return asked


# read what has been asked
def asked_questions(
    project, limit: Optional[int] = None
) -> List[AskedQuestion]:
    '''
    Args:
        project (Project): The project.
        limit (int, optional): Return only the most recent this many.

    Returns:
        List[AskedQuestion]: Questions in the order they were asked. A line \
            that cannot be parsed is skipped rather than failing the read — \
            a truncated last line should not hide the history above it.
    '''
    path = questions_file(project)
    if not path.is_file():
        return []

    found: List[AskedQuestion] = []
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as error:
        log.warning('could not read the question log: %s', error)

        return []

    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('text'):
            found.append(AskedQuestion(
                text=str(row['text']), asked_at=str(row.get('asked_at', ''))
            ))

    return found[-limit:] if limit else found
