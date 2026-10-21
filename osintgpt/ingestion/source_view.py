'''Provider-free source reads with explicit current-file and saved-text identity.'''

from .snapshots import load_snapshot, read_snapshot, source_fingerprint, transcriptions_current


def reference_for(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def reference_parts(ref, root, covered):
    '''Resolve only the file portion; record identities may contain slashes or #.'''
    boundaries = [len(ref)] + [index for index in range(len(ref) - 1, -1, -1) if ref[index] == '#']
    for boundary in boundaries:
        candidate = (root / ref[:boundary]).resolve()
        if candidate in covered:
            return candidate, ref[boundary:]
    return None, ''


def source_state(path, mapping, snapshot, cache_dir=None):
    try:
        fingerprint = source_fingerprint(path, mapping)
    except FileNotFoundError:
        return 'missing'
    except OSError:
        return 'unreadable'
    if fingerprint != snapshot.fingerprint:
        return 'changed'
    if cache_dir and not transcriptions_current(snapshot, cache_dir):
        return 'transcription_changed'
    return 'unchanged'


def read_source(project, path, ref, source_version=''):
    from .sources import Corpus

    root = project.paths.root
    mapping = Corpus.load(project.paths.sources).mapping_for(path, root)
    source_ref = reference_for(path, root)
    if source_version:
        snapshot = read_snapshot(project.paths.extracts, source_version)
        if snapshot.ref != source_ref:
            raise ValueError('The saved text version belongs to a different source.')
    else:
        snapshot = load_snapshot(path, mapping, ref=source_ref, cache_dir=project.paths.extracts)
    _, record = reference_parts(ref, root, {path.resolve()})
    if record:
        document = snapshot.document(source_ref + record)
        if document is None:
            raise ValueError('Record does not exist in this text version.')
        documents = [document]
    else:
        documents = snapshot.documents
    parts, records, offset = [], [], 0
    for document in documents:
        prefix = f'[Record {document.ref}]\n' if len(documents) > 1 else ''
        part = prefix + document.text
        records.append({'ref': document.ref, 'start': offset + len(prefix),
                        'end': offset + len(part),
                        'locations': [span.__dict__ for span in document.spans]})
        parts.append(part)
        offset += len(part) + 2
    return '\n\n'.join(parts), {
        'source_ref': source_ref, 'source_version': snapshot.version,
        'source_fingerprint': snapshot.fingerprint,
        'source_state': source_state(path, mapping, snapshot, project.paths.extracts),
        'text_basis': 'saved_snapshot' if source_version else 'current_source',
        'missing_pages': snapshot.missing_pages, 'records': records,
    }
