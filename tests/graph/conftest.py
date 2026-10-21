from types import SimpleNamespace

import pytest

from osintgpt import Project
from osintgpt.graph import Edge, Entity, graph_for
from osintgpt.ingestion import Corpus
from osintgpt.ingestion.snapshots import load_snapshot


@pytest.fixture
def graph_case(tmp_path):
    project = Project.create('Retrieval', home=tmp_path).with_settings(graph_enabled=True)

    def publish(ref, text, source='Alpha', target='Beta', quotes=None, **fields):
        path = project.paths.root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
        Corpus.load(project.paths.sources).register(ref)
        snapshot = load_snapshot(path, ref=ref, cache_dir=project.paths.extracts)
        document = snapshot.document(ref)
        quotes = tuple(quotes or [text])
        provenance = snapshot.provenance(document, quotes[0])
        provenance['supporting'] = [{'quote': quote, 'matches': snapshot.provenance(document, quote)['matches']}
                                   for quote in quotes]
        fields.setdefault('source_id', source.casefold())
        fields.setdefault('target_id', target.casefold())
        edge = Edge(source, target, fields.pop('relation', 'funds'), ref, quotes[0],
                    provenance=provenance, evidence_quotes=quotes, **fields)
        with graph_for(project) as graph:
            graph.add([Entity(edge.source_id, source), Entity(edge.target_id, target)], [edge])
        return edge

    return SimpleNamespace(project=project, publish=publish)
