'''Synthetic cosine graph and accounting provider shared by Snowball adapters.'''

import math

from osintgpt.llm.base import EmbeddingProvider, EmbeddingPurpose
from osintgpt.llm.usage import Usage
from osintgpt.vector_store import SearchResult, StoredChunk


def vector(angle):
    return [math.cos(angle), math.sin(angle)]


class GraphEmbedder(EmbeddingProvider):
    model = 'text-embedding-3-small'

    def __init__(self, count=30, recorder=None, report=True):
        self.vectors = {'start': vector(-0.1)}
        self.vectors.update({f'passage {i}': vector(i * 0.03) for i in range(count)})
        self.calls = []
        self.recorder = recorder
        self.report = report

    def embed(self, texts, *, purpose=EmbeddingPurpose.DOCUMENT):
        self.calls.append((list(texts), purpose))
        if self.report:
            self._record(Usage('fake', self.model, input_tokens=7))
        return [self.vectors[' '.join(text.casefold().split())] for text in texts]


class GraphStore:
    def __init__(self, count=8, copies=0):
        self.chunks = [StoredChunk(
            'posts.jsonl', i, f'passage {i}', GraphEmbedder.model,
            document_ref=f'posts.jsonl#{i}',
            timestamp=['2026-02-03T12:00:00Z', '2026-01-01', 'bad date', ''][i % 4],
            metadata={'published': '28/10/2026'},
        ) for i in range(count)]
        self.vectors = [vector(i * 0.03) for i in range(count)]
        for i in range(copies):
            self.chunks.append(StoredChunk(
                'copies.jsonl', i, ' PASSAGE   0 ', GraphEmbedder.model,
                document_ref=f'copies.jsonl#{i}',
            ))
            self.vectors.append(vector(0))
        self.requests = []
        self.closed = False

    def search(self, query, embedding_model, top_k=10, refs=None):
        self.requests.append((list(query), embedding_model, top_k, refs))
        qnorm = math.sqrt(sum(v * v for v in query))
        results = [SearchResult(chunk, sum(a * b for a, b in zip(query, v)) / qnorm)
                   for chunk, v in zip(self.chunks, self.vectors)
                   if chunk.embedding_model == embedding_model and (refs is None or chunk.ref in refs)]
        return sorted(results, key=lambda result: -result.score)[:top_k]

    def close(self):
        self.closed = True
