# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: mapping.py
# Description: Assigning field roles to structured files, and showing what a
#   mapping would keep before it is committed to.
# =================================================================================

# type hints
from typing import Any, Dict, List, Optional, Tuple

# import osintgpt ingestion
from osintgpt.ingestion import FieldMapping, describe_fields, preview_files
from osintgpt.ingestion.loaders import needs_mapping

# Counts already computed, keyed by the files and the roles that decide what
# survives. See `_counted`.
_COUNTS: Dict[tuple, Any] = {}

# Enough for the mappings one operator tries against a folder.
MAX_COUNTS = 32


# structured files under a folder, grouped by the fields they share
def schema_groups(folder) -> List[Tuple[List, Dict[str, dict]]]:
    '''
    Collect the files still needing roles, one entry per distinct schema.

    Args:
        folder (Path): Directory to scan.

    Returns:
        List[Tuple[List, Dict[str, dict]]]: Per schema, the files carrying it \
            and the field description of the first of them.
    '''
    groups: Dict[tuple, Tuple[List, Dict[str, dict]]] = {}

    for path in sorted(folder.rglob('*')):
        if not path.is_file():
            continue
        described = described_fields(path)
        if not described:
            continue
        paths, _ = groups.setdefault(tuple(described), ([], described))
        paths.append(path)

    return list(groups.values())


# ask which field plays which role for one schema
def assign_roles(st, paths, described, key: str) -> Optional[FieldMapping]:
    '''
    Args:
        st: The Streamlit module.
        paths (List[Path]): Files sharing this schema.
        described (Dict[str, dict]): Field descriptions to choose from.
        key (str): Prefix for this schema's widgets. Two of these can be on \
            one page, and widgets sharing a key share a value.

    Returns:
        Optional[FieldMapping]: The mapping, or None when no content field \
            was named.
    '''
    fields = list(described)
    label = (
        paths[0].name if len(paths) == 1
        else f'{paths[0].name} and {len(paths) - 1} more like it'
    )

    with st.expander(f'Fields in {label}', expanded=True):
        st.caption(
            'Only content is embedded and searched. The rest travel with the '
            'document, and timestamp is what a date filter reads.'
        )
        st.dataframe(
            [
                {
                    'field': name,
                    'filled': f'{facts["filled"]}/{facts["sampled"]}',
                    'avg chars': facts['average_length'],
                    'unique': facts['unique'],
                    'example': facts['example']
                }
                for name, facts in described.items()
            ],
            hide_index=True,
            width='stretch'
        )

        primary = st.selectbox(
            'Content — the body', [''] + fields, key=f'{key}-primary',
            help='A record whose content field is empty is skipped.'
        )
        extra = st.multiselect(
            'Also embed', [f for f in fields if f != primary],
            key=f'{key}-extra',
            help='Joined after the body. Identifiers do not belong here — '
                 'they repeat across every record and blur the embedding.'
        )
        timestamp = st.selectbox(
            'Timestamp', [''] + fields, key=f'{key}-timestamp'
        )
        author = st.selectbox('Author', [''] + fields, key=f'{key}-author')
        identity = st.selectbox(
            'Identity', [''] + fields, key=f'{key}-identity',
            help='Makes the citation, and keeps a re-index updating a record '
                 'rather than duplicating it.'
        )
        metadata = st.multiselect(
            'Metadata', fields, key=f'{key}-metadata',
            help='Kept with the document, never embedded.'
        )
        min_chars = st.number_input(
            'Minimum characters', min_value=0, value=0, step=10,
            key=f'{key}-min-chars',
            help='Records with less content than this are skipped. Catches '
                 'signature-only and placeholder rows.'
        )

        if not primary:
            return None

        mapping = FieldMapping(
            content=(primary, *extra),
            metadata=tuple(metadata),
            timestamp=timestamp,
            author=author,
            identity=identity,
            min_chars=int(min_chars)
        )
        _outcome(st, paths, mapping)

    return mapping


def _outcome(st, paths, mapping: FieldMapping) -> None:
    '''
    What this mapping would keep, shown while it can still change the choice.
    '''
    label = paths[0].name if len(paths) == 1 else f'{len(paths)} files'
    try:
        with st.spinner(f'Reading {label}…'):
            preview = _counted(paths, mapping)
    except Exception as error:  # noqa: BLE001 — a view, not a pass
        st.caption(f'Could not read {label}: {error}')

        return

    st.caption(preview.summary)
    if preview.dropped and not preview.kept:
        st.error(
            'Every record would be skipped. Check that the content field is '
            'the body of the record.'
        )
    if preview.example:
        # A display element rather than a disabled text area: a keyed widget
        # reads its value from session state and would keep showing the first
        # mapping's example after the fields changed.
        st.caption('Shortest record that would be kept')
        st.code(preview.example, language=None, wrap_lines=True)


def _counted(paths, mapping: FieldMapping):
    '''
    Hold each count against the roles that can change it.

    Every widget interaction reruns the script, and reading the files again
    each time would make an accurate count unusable. Only content and the
    floor decide what survives, so choosing a timestamp or an author reuses
    what is already known.
    '''
    key = (
        tuple(_stamp(path) for path in paths),
        mapping.content,
        mapping.min_chars
    )

    if key not in _COUNTS:
        if len(_COUNTS) >= MAX_COUNTS:
            _COUNTS.clear()
        _COUNTS[key] = preview_files(paths, mapping)

    return _COUNTS[key]


def _stamp(path) -> tuple:
    stat = path.stat()

    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


# the fields of a structured file, or nothing for anything else
def described_fields(path) -> Dict[str, dict]:
    '''
    Args:
        path (Path): File to describe.

    Returns:
        Dict[str, dict]: Field descriptions, empty when the file needs no \
            mapping or cannot be read.
    '''
    if not needs_mapping(path):
        return {}

    try:
        return describe_fields(path)
    except Exception:  # noqa: BLE001 — a view, not a pass
        return {}
