'''Publication preserves claims while rechecking frozen cross-source decisions.'''

from dataclasses import replace

from osintgpt.graph import GraphStore, extract_document
from osintgpt.graph.publication import validate_identities
from osintgpt.ingestion.snapshots import TextSnapshot
from test_interactive_identity import prior_entity, native_lookup, NOW
from test_extraction_quality import Script


def test_missing_candidate_downgrades_saved_reuse_without_discarding_paid_claims():
    prior = prior_entity()
    provider, requests = native_lookup('responses', prior.key)
    try:
        extraction = extract_document(provider, 'new.md', NOW, known_entities=[prior])
    finally:
        provider.client.close()
    snapshot = TextSnapshot('new.md', 'hash', extraction.entities[0].provenance['source_version'], ())
    with GraphStore(':memory:') as graph:
        graph.add([prior], [])
        accepted, claims, warnings = validate_identities(graph, snapshot, extraction.entities, extraction.edges)
        assert not warnings and claims[0].source_id == prior.key
        graph.forget(['past.md'])
        separate, claims, warnings = validate_identities(graph, snapshot, extraction.entities, extraction.edges)
        assert len(warnings) == 1 and len(claims) == 1 and claims[0].source_id != prior.key
        assert claims[0].evidence == extraction.edges[0].evidence and len(requests) == 3
        assert all(mention['source_ref'] == 'new.md' for entity in separate
                   for mention in entity.provenance['mentions'])
        assert separate[0].provenance['resolution']['action'] == 'unresolved'


def test_refresh_local_id_changes_do_not_rename_supported_identity():
    data = {'entities': [{'id': 'first', 'name': 'Orion', 'evidence': ['Orion builds Prism.']}], 'edges': []}
    before = extract_document(Script(data), 'source.md', 'Orion builds Prism.')
    data['entities'][0]['id'] = 'different-model-id'
    after = extract_document(Script(data), 'source.md', 'Orion builds Prism.\nNew appendix.')
    assert before.entities[0].key == after.entities[0].key
    assert before.entities[0].provenance['source_version'] != after.entities[0].provenance['source_version']
