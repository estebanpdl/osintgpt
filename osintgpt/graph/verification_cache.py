'''Query-local evidence caches guarded by source content and registration versions.'''

import re
import stat

from osintgpt.ingestion import Corpus
from osintgpt.ingestion.documents import FieldMapping
from osintgpt.ingestion.snapshots import source_fingerprint
from osintgpt.ingestion.transcription import cache_path, cached_transcriber
from osintgpt.projects.paths import CANON_DIR


class VerificationChanged(ValueError):
    pass


def _signature(path, mapping=None):
    try:
        return ('readable', source_fingerprint(path, mapping))
    except OSError as error:
        return (type(error).__name__, error.errno)


def _file_state(path):
    try:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError as error:
        return type(error).__name__, error.errno


def _mapping_state(path):
    '''Directory membership and path identity; source contents are hashed separately.'''
    try:
        info = path.lstat()
        identity = info.st_dev, info.st_ino, info.st_mode
        if stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            return (*identity, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        return identity
    except OSError as error:
        return type(error).__name__, error.errno


class EvidenceCache:
    '''Reuse extraction/quotation checks within one fixed verification state.

    Source content is hashed on validation, including same-size edits. Immutable
    saved-text files are watched for replacement; cached transcription contents
    are also checked. A change invalidates the query rather than mixing versions.
    '''

    def __init__(self, project):
        self.project = project
        self.registration = _signature(project.paths.sources)
        self.corpus = Corpus.load(project.paths.sources)
        self.documents, self.saved, self.states, self.results = {}, {}, {}, {}
        self.sources, self.files, self.transcriptions = {}, {}, {}
        self.registration_policy = None
        self.mappings = None
        self.mapping_paths, self.path_mappings = {}, {}
        self._transcriber = cached_transcriber(project.paths.extracts)

    def _watch_mapping_path(self, path):
        self.mapping_paths.setdefault(path, _mapping_state(path))
        # Watch both names for symbolic links/junctions, including on Python 3.10.
        resolved = path.resolve()
        if resolved != path:
            self.mapping_paths.setdefault(resolved, _mapping_state(resolved))

    def mapping_for(self, path):
        if self.mappings is None:
            self._watch_mapping_path(self.project.paths.root / CANON_DIR)
            self.mappings = self.corpus.mappings(self.project.paths.root, watch=self._watch_mapping_path)
        if path not in self.path_mappings:
            self.path_mappings[path] = self.mappings.get(path.resolve(), FieldMapping())
        return self.path_mappings[path]

    def bind_policy(self, require_registered):
        if self.registration_policy is not None and self.registration_policy != require_registered:
            raise ValueError('Evidence cache cannot change source-registration policy.')
        self.registration_policy = require_registered

    def watch_source(self, ref, path, mapping, readable):
        self.sources.setdefault(ref, (path, mapping, _signature(path, mapping) if readable else None))

    def watch_saved(self, version):
        if not re.fullmatch(r'[0-9a-f]{64}', version):
            return
        path = self.project.paths.extracts / 'text' / f'{version}.json'
        self.files.setdefault(path, _file_state(path))

    def watch_transcriptions(self, snapshot):
        if snapshot:
            for name in snapshot.transcriptions:
                if not re.fullmatch(r'[0-9a-f]{64}\.md', name):
                    continue
                path = self.project.paths.extracts / name
                self._watch_transcription(path)

    def _watch_transcription(self, path):
        if path not in self.transcriptions:
            self.transcriptions[path] = _signature(path)

    def read_transcription(self, image, page):
        # Also watch cache misses: a later local transcription changes current text.
        self._watch_transcription(cache_path(self.project.paths.extracts, image))
        return self._transcriber(image, page)

    def problem(self):
        if _signature(self.project.paths.sources) != self.registration:
            return 'Source registrations or field mappings changed.'
        if any(_mapping_state(path) != signature for path, signature in self.mapping_paths.items()):
            return 'Source paths or registered folder membership changed.'
        for ref, (path, mapping, signature) in self.sources.items():
            if signature is not None and _signature(path, mapping) != signature:
                return f'Source content or availability changed: {ref}.'
        if any(_file_state(path) != signature for path, signature in self.files.items()):
            return 'A saved text snapshot changed or became unavailable.'
        if any(_signature(path) != signature for path, signature in self.transcriptions.items()):
            return 'Cached source transcription changed or became unavailable.'
        return ''

    def check(self):
        if problem := self.problem():
            raise VerificationChanged(problem + ' Run a new exploration to refresh evidence checks.')

    def metadata(self):
        return {'basis': 'query_snapshot', 'sources': [
            {'source_ref': ref, 'content_signature': signature,
             'current_version': self.documents[ref][0].version if self.documents.get(ref, (None,))[0] else '',
             'saved_versions': sorted(version for source, version in self.states if source == ref)}
            for ref, (_, _, signature) in sorted(self.sources.items())]}
