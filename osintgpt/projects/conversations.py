# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: conversations.py
# Description: What has been asked and answered, kept per conversation so it
#   survives a refresh. Append-only: nothing ever rewrites a file holding
#   history.
# =================================================================================

# import modules
import json
import logging
import re

# import submodules
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# type hints
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger('osintgpt.projects')

CONVERSATIONS_DIR = 'conversations'
SUFFIX = '.jsonl'

META = 'meta'
TURN = 'turn'

# Record types this version understands. An unknown one is skipped rather than
# refused, so a transcript written by a newer osintgpt still opens here.
KNOWN = {META, TURN}

# A cheap test for "this line might be a header", so listing does not decode
# every answer. It filters rather than decides: a question quoting the same
# text passes it and then fails the parse.
MARKER = f'"type": "{META}"'

# Characters of the first question kept in the filename. Nothing parses the
# slug back — identity is the timestamp — so it may be lossy, and Windows'
# 260-character path limit is what keeps it short.
SLUG_CHARS = 48

# How much of the first question becomes the title when none was set.
TITLE_CHARS = 60

# Pairs of question and answer carried into the next question by default. A
# window rather than the whole thread, so retrieved passages stay the dominant
# context and the answer stays grounded in the corpus.
WINDOW = 3

UNTITLED = 'Untitled conversation'

# Stamped into filenames. Colons are legal in a path on POSIX and not on
# Windows, so the ISO form is not usable as written.
STAMP = '%Y-%m-%dT%H-%M-%S'


# Turn class
@dataclass(frozen=True)
class Turn:
    '''
    One question and the answer it got.
    '''
    question: str
    # The answer as plain data, as `agentic.answer_to_dict` wrote it. Opaque
    # here on purpose: this module holds project state and never has to change
    # when the shape of an answer does.
    answer: Dict[str, Any] = field(default_factory=dict)
    asked_at: str = ''


# Conversation class
@dataclass
class Conversation:
    '''
    One thread of questions against a project, and the file holding it.
    '''
    path: Path
    title: str = ''
    created_at: str = ''
    turns: List[Turn] = field(default_factory=list)

    @property
    def id(self) -> str:
        '''
        Returns:
            str: The filename without its suffix. Stable for the life of the \
                conversation, because a rename appends rather than moves.
        '''
        return self.path.stem

    @property
    def name(self) -> str:
        '''
        Returns:
            str: What to show in a list.
        '''
        return self.title or UNTITLED

    @property
    def updated_at(self) -> float:
        '''
        Returns:
            float: When the file was last written, or 0.0 when it does not \
                exist yet. Taken from the filesystem rather than recorded, so \
                nothing has to be kept in step with it.
        '''
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def __len__(self) -> int:
        return len(self.turns)


# where a project keeps its conversations
def conversations_dir(project) -> Path:
    '''
    Args:
        project (Project): The project.

    Returns:
        Path: Its conversations directory. Not created.
    '''
    return Path(project.paths.root) / CONVERSATIONS_DIR


# start a conversation
def start_conversation(project, question: str) -> Conversation:
    '''
    Create a conversation named after the question that opened it.

    Nothing is written until there is a turn to write, so a thread the
    operator opened and abandoned leaves no file behind.

    Args:
        project (Project): The project being asked.
        question (str): The first question, which names the file.

    Returns:
        Conversation: An empty conversation, not yet on disk.
    '''
    moment = datetime.now(timezone.utc)
    slug = slug_for(question)
    stem = f'{moment.strftime(STAMP)}-{slug}' if slug else moment.strftime(STAMP)

    return Conversation(
        path=_free_path(conversations_dir(project), stem),
        title=title_for(question),
        created_at=moment.isoformat(timespec='seconds')
    )


# add a turn
def append_turn(
    conversation: Conversation, question: str, answer: Dict[str, Any]
) -> Optional[Turn]:
    '''
    Record one exchange, writing the header first if the file is new.

    Failing to record is not failing to answer: a read-only directory should
    cost the transcript, not the reply the analyst is reading.

    Args:
        conversation (Conversation): The conversation to add to.
        question (str): The question, as asked.
        answer (Dict[str, Any]): The answer as plain data.

    Returns:
        Optional[Turn]: What was recorded, or None if it could not be.
    '''
    turn = Turn(
        question=question,
        answer=answer,
        asked_at=datetime.now(timezone.utc).isoformat(timespec='seconds')
    )

    records = []
    if not conversation.path.exists():
        records.append({
            'type': META,
            'title': conversation.title,
            'created_at': conversation.created_at
        })
    records.append({
        'type': TURN,
        'question': turn.question,
        'answer': turn.answer,
        'asked_at': turn.asked_at
    })

    if not _append(conversation.path, records):
        return None

    conversation.turns.append(turn)

    return turn


# rename a conversation
def rename_conversation(conversation: Conversation, title: str) -> bool:
    '''
    Set a new title by appending one, never by rewriting the file.

    The last header wins on read. A rename that rewrote the transcript would
    put every answer in it at risk to change one line.

    Args:
        conversation (Conversation): The conversation.
        title (str): The new title.

    Returns:
        bool: True when it was written.
    '''
    cleaned = (title or '').strip()
    if not cleaned or cleaned == conversation.title:
        return False

    # Nothing on disk yet: the title is still only in memory, and the header
    # that has not been written will carry it.
    if not conversation.path.exists():
        conversation.title = cleaned

        return True

    if not _append(conversation.path, [{'type': META, 'title': cleaned}]):
        return False

    conversation.title = cleaned

    return True


# forget a conversation
def delete_conversation(conversation: Conversation) -> bool:
    '''
    Args:
        conversation (Conversation): The conversation to remove.

    Returns:
        bool: True when a file was removed.
    '''
    try:
        conversation.path.unlink()
    except FileNotFoundError:
        return False
    except OSError as error:
        log.warning('could not delete the conversation: %s', error)

        return False

    return True


# read one conversation
def load_conversation(path) -> Conversation:
    '''
    Read a transcript.

    A line that cannot be parsed is skipped rather than failing the read: a
    truncated last line must not hide the conversation above it.

    Args:
        path (Union[str, Path]): The transcript file.

    Returns:
        Conversation: What the file holds, empty when it cannot be read.
    '''
    path = Path(path)
    conversation = Conversation(path=path)

    for row in _records(path):
        kind = row.get('type')
        if kind == META:
            # The last header wins, and a rename writes only the field it
            # changes, so an absent one keeps what came before.
            if row.get('title'):
                conversation.title = str(row['title'])
            if row.get('created_at'):
                conversation.created_at = str(row['created_at'])
        elif kind == TURN and row.get('question'):
            answer = row.get('answer')
            conversation.turns.append(Turn(
                question=str(row['question']),
                answer=answer if isinstance(answer, dict) else {},
                asked_at=str(row.get('asked_at', ''))
            ))

    return conversation


# every conversation a project holds
def list_conversations(project) -> List[Conversation]:
    '''
    The project's conversations, most recently written first.

    Args:
        project (Project): The project.

    Returns:
        List[Conversation]: Conversations without their turns.
    '''
    return list_in(conversations_dir(project))


# every conversation in a directory
def list_in(directory) -> List[Conversation]:
    '''
    Read the headers of every transcript in a directory.

    Takes a directory rather than a project so a caller can cache on what the
    directory looks like without having to hash a project to do it.

    Args:
        directory (Union[str, Path]): A conversations directory.

    Returns:
        List[Conversation]: Conversations without their turns, most recently \
            written first. Only headers are parsed, so listing a long history \
            does not cost the decoding of every answer in it.
    '''
    directory = Path(directory)
    if not directory.is_dir():
        return []

    found = []
    for path in directory.glob(f'*{SUFFIX}'):
        if path.is_file():
            found.append(_header(path))

    return sorted(found, key=lambda c: c.updated_at, reverse=True)


# find a conversation by id
def open_conversation(project, conversation_id: str) -> Optional[Conversation]:
    '''
    Args:
        project (Project): The project.
        conversation_id (str): The id, as `Conversation.id` reports it.

    Returns:
        Optional[Conversation]: The conversation with its turns, or None when \
            no such file exists.
    '''
    if not conversation_id or '/' in conversation_id or '\\' in conversation_id:
        return None

    path = conversations_dir(project) / f'{conversation_id}{SUFFIX}'

    return load_conversation(path) if path.is_file() else None


# the last few exchanges, for the next question
def recent_pairs(
    conversation: Optional[Conversation], size: int = WINDOW
) -> List[Tuple[str, str]]:
    '''
    The sliding window a follow-up is asked with.

    Whole pairs only, which is why the size counts pairs and not characters: a
    truncated answer is something a model reasons from as though it were
    complete. Passages are not carried — the tools retrieve them again.

    The one place this module looks inside a stored answer.

    Args:
        conversation (Conversation, optional): The thread so far, or None.
        size (int): Pairs to carry. Zero or less carries none.

    Returns:
        List[Tuple[str, str]]: Question and answer, oldest first. Empty for \
            the first question in a thread, which has nothing to build on.
    '''
    if conversation is None or size <= 0:
        return []

    pairs = []
    for turn in conversation.turns[-size:]:
        answer = turn.answer.get('answer', '') if turn.answer else ''
        if turn.question and answer:
            pairs.append((turn.question, str(answer)))

    return pairs


# a filename-safe fragment of the opening question
def slug_for(question: str) -> str:
    '''
    Cut at a word boundary rather than mid-word, so a long question still
    yields a readable name.

    Args:
        question (str): The opening question.

    Returns:
        str: The slug, empty when nothing usable survives — a question in a \
            non-Latin script leaves nothing. The filename is then the \
            timestamp alone.
    '''
    slug = re.sub(r'[^a-z0-9]+', '-', (question or '').lower()).strip('-')
    if len(slug) <= SLUG_CHARS:
        return slug

    cut = slug[:SLUG_CHARS]
    # Keep whole words, unless the first one is longer than the whole budget.
    if '-' in cut:
        cut = cut[:cut.rindex('-')]

    return cut.strip('-')


# a readable name for the conversation
def title_for(question: str) -> str:
    '''
    Args:
        question (str): The opening question.

    Returns:
        str: The question, shortened for a list. The full text is always the \
            first turn, so nothing is lost by cutting here.
    '''
    cleaned = ' '.join((question or '').split())
    if len(cleaned) <= TITLE_CHARS:
        return cleaned

    cut = cleaned[:TITLE_CHARS]
    if ' ' in cut:
        cut = cut[:cut.rindex(' ')]

    return cut.rstrip(' ,.;:') + '…'


def _header(path: Path) -> Conversation:
    '''
    A conversation's title and creation time, without parsing its turns.

    Headers are appended rather than rewritten, so the file is read to its
    end — a renamed conversation carries its newest title last. Only lines
    that could be a header are parsed: a listing must not cost the JSON
    decoding of every answer in every conversation, and the answers are by
    far the bulk of the bytes.
    '''
    conversation = Conversation(path=path)

    for line in _lines(path):
        if MARKER not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get('type') != META:
            continue
        if row.get('title'):
            conversation.title = str(row['title'])
        if row.get('created_at'):
            conversation.created_at = str(row['created_at'])

    return conversation


def _lines(path: Path) -> List[str]:
    try:
        return path.read_text(encoding='utf-8').splitlines()
    except OSError as error:
        log.warning('could not read the conversation: %s', error)

        return []


def _records(path: Path) -> List[Dict[str, Any]]:
    '''
    The file's records, skipping anything unreadable.
    '''
    found = []
    for line in _lines(path):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('type') in KNOWN:
            found.append(row)

    return found


def _append(path: Path, records: List[Dict[str, Any]]) -> bool:
    '''
    Add records to a transcript, one JSON object per line.
    '''
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, default=str) + '\n'
                )
    except OSError as error:
        log.warning('could not record the conversation: %s', error)

        return False

    return True


def _free_path(directory: Path, stem: str) -> Path:
    '''
    A path nothing occupies. Two conversations opened in the same second with
    the same first question would otherwise be one file.
    '''
    candidate = directory / f'{stem}{SUFFIX}'
    suffix = 2
    while candidate.exists():
        candidate = directory / f'{stem}-{suffix}{SUFFIX}'
        suffix += 1

    return candidate
