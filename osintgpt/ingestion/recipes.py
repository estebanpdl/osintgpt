'''Index recipes and conservative reconciliation of older completion records.'''

import hashlib
import json
from dataclasses import replace

from . import chunking
from .images import is_image
from .indexing import _ref_for, content_hash
from .loaders import load_documents
from osintgpt.vector_store.records import StoredChunk


# Bump when extraction, record filtering or provenance semantics change.
INGESTION_VERSION = 1


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(',', ':')
    ).encode('utf-8')).hexdigest()


def implementation(value) -> str:
    return f'{type(value).__module__}.{type(value).__qualname__}'


def recipe_for(mapping, embedder, *, transcribes=False) -> str:
    options = getattr(embedder, 'indexing_options', lambda: {})()
    return digest({
        'version': 2,
        'ingestion_version': INGESTION_VERSION,
        'mapping': mapping.to_dict(),
        'chunking': {
            'version': chunking.CHUNKING_VERSION,
            'max_chars': chunking.MAX_CHARS,
            'min_content_room': chunking.MIN_CONTENT_ROOM,
            'breadcrumb_separator': chunking.BREADCRUMB_SEPARATOR
        },
        'embedding': {
            'implementation': implementation(embedder),
            'provider': getattr(embedder, 'provider', ''),
            'model': embedder.model,
            'purpose': 'document',
            'options': options
        },
        'transcription_enabled': transcribes
    })


def destination_for(store, root) -> str:
    return digest({
        'implementation': implementation(store),
        'destination': store.indexing_destination(root)
    })


def supports_legacy_checkpoint(embedder) -> bool:
    '''Only adopt v1 jobs whose unrecorded options have known v1 defaults.

    A custom provider may have options v1 never hashed. Matching its class
    and model alone is not sufficient evidence to reuse its old vectors.
    '''
    name = implementation(embedder)
    options = getattr(embedder, 'indexing_options', lambda: {})()
    if name == 'osintgpt.llm.openai_compat.OpenAICompatEmbedding':
        return options == {
            'endpoint': str(getattr(embedder.client, 'base_url', '')).rstrip('/'),
            'request_version': 1
        }
    if name == 'osintgpt.llm.gemini.GeminiEmbedding':
        return (
            options.get('endpoint', '').rstrip('/') in (
                '', 'https://generativelanguage.googleapis.com'
            )
            and options.get('api_version') in ('', 'v1beta')
            and options == {
                'endpoint': options.get('endpoint'),
                'api_version': options.get('api_version'),
                'task_style': embedder.task_style,
                'document_task': 'RETRIEVAL_DOCUMENT'
                    if embedder.task_style == 'task_type' else 'title: none | text: {text}',
                'request_version': 1
            }
        )
    return False


def prepare_chunks(path, ref, mapping, embedder, transcriber=None, *, statistics=None, record_ends=None):
    '''Derive the exact payload, keeping both file and record identity.'''
    chunks, texts, document_refs = [], [], []
    documents = load_documents(path, mapping, transcriber, statistics=statistics)
    if statistics is not None:
        statistics['documents'] = len(documents)
    for document in documents:
        start = len(chunks)
        # Project-local record refs remain usable when the project is moved.
        source_ref = path.as_posix()
        document_ref = document.ref
        if document_ref == source_ref or document_ref.startswith(source_ref + '#'):
            document_ref = ref + document_ref[len(source_ref):]
        for piece in chunking.chunk_document(document.text, max_chars=chunking.MAX_CHARS):
            texts.append(piece.rendered)
            # Kept separately for compatibility with the first journal format.
            document_refs.append(document.ref)
            chunks.append(StoredChunk(
                ref=ref, sequence=len(chunks), text=piece.text,
                embedding_model=embedder.model, path=piece.path,
                timestamp=document.timestamp, author=document.author,
                metadata=dict(document.metadata), document_ref=document_ref
            ))
        if record_ends is not None and len(chunks) > start:
            record_ends.append(len(chunks))
    if statistics is not None:
        statistics['chunked_documents'] = len(record_ends) if record_ends is not None else len(documents)
    return chunks, texts, document_refs


def reconcile_plan(plan, state, root, embedder, store, mappings, recipes,
                   destination, pending):
    '''Verify completion against the destination without embedding anything.

    A legacy OpenAI default is inferred only after every stored chunk matches
    current extraction. Other legacy providers/options need an explicit rebuild.
    No transcriber is used during this audit, so it cannot trigger a paid call.
    '''
    inventory = store.index_inventory() if plan.unchanged else {}
    changed, unchanged, adopted, blocked = list(plan.changed), [], [], []
    for path in plan.unchanged:
        ref = _ref_for(path, root)
        known = state.documents[ref]
        if ref in pending:
            changed.append(path)
            continue
        expected = {embedder.model: known.chunks} if known.chunks else {}
        present = inventory.get(ref, {})
        if known.recipe and known.destination and known.embedding_model:
            if present != expected or known.embedding_model != embedder.model:
                changed.append(path)
            else:
                state.documents[ref] = replace(known, embedding_provider=getattr(embedder, 'provider', '') or ('gemini' if type(embedder).__name__ == 'GeminiEmbedding' else ''))
                unchanged.append(path)
            continue
        compatible = False
        try:
            may_adopt = getattr(embedder, 'can_adopt_legacy_index', lambda: False)()
            if may_adopt and not is_image(path) and present == expected:
                derived, _, _ = prepare_chunks(path, ref, mappings[ref], embedder)
                stored = store.chunks_for(ref)
                compatible = (
                    len(derived) == known.chunks
                    and [replace(c, document_ref='') for c in stored]
                    == [replace(c, document_ref='') for c in derived]
                    and known.hash == content_hash(path.read_bytes())
                )
        except Exception:
            # Unknown extraction/storage is not evidence of compatibility.
            compatible = False
        if compatible:
            state.documents[ref] = replace(
                known, recipe=recipes[ref], destination=destination,
                embedding_model=embedder.model,
                embedding_provider=getattr(embedder, 'provider', '') or ('gemini' if type(embedder).__name__ == 'GeminiEmbedding' else '')
            )
            adopted.append(ref)
            unchanged.append(path)
        else:
            blocked.append((ref,
                'Legacy index configuration cannot be verified against the current '
                'model, mapping and destination. Existing chunks were preserved. '
                'Run osintgpt index --force with the intended configuration to rebuild.'
            ))
    return replace(plan, changed=changed, unchanged=unchanged), adopted, blocked
