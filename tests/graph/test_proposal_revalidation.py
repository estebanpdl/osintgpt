'''Local correction never widens old evidence scope or requires provider calls.'''

from dataclasses import replace

from osintgpt.graph import graph_for
from osintgpt.graph.records import RejectedClaim
from osintgpt.graph.revalidation import revalidate_proposals
from test_identity_review import pair


def test_local_preview_apply_and_repeat_preserve_original_audit(pair):
    project, first, _ = pair
    edge = first.edges[0]
    candidate = {'source': '0', 'target': '1', 'relation': 'supports', 'evidence': ['**Orion** supports Lyra.']}
    rejected = RejectedClaim(edge.ref, edge.provenance['source_ref'], edge.provenance['source_version'], 1,
                             candidate, 'Original exact quotation mismatch.', supplied_ranges=((0, len(edge.evidence)),))
    with graph_for(project) as graph:
        graph.record_rejections([rejected, replace(rejected, candidate={**candidate, 'evidence': ['Orion opposes Lyra.']})])
    path = project.paths.root / 'graph.sqlite'
    original = path.read_bytes()
    preview = revalidate_proposals(project)
    assert path.read_bytes() == original and preview['provider_calls'] == 0
    assert len(preview['accepted_claims']) == 1
    assert [item['status'] for item in preview['outcomes']] == ['accepted_locally', 'rejected']
    page = revalidate_proposals(project, limit=1)
    assert page['truncated'] and page['next_offset'] == 1
    next_page = revalidate_proposals(project, offset=1, limit=1)
    assert next_page['outcomes'][0]['status'] == 'rejected' and not next_page['truncated']
    applied = revalidate_proposals(project, apply=True, expected_revision=preview['revision'])
    assert applied['status'] == 'applied' and applied['backup']
    with graph_for(project) as graph:
        assert graph.edge_count == 3
        assert graph.rejections()[0].reason == 'Original exact quotation mismatch.'
        assert graph.rejections()[0].status == 'accepted_locally'
    assert not revalidate_proposals(project)['accepted_claims']


def test_old_audit_without_ranges_is_explained_not_guessed(pair):
    project, first, _ = pair
    edge = first.edges[0]
    rejected = RejectedClaim(edge.ref, edge.provenance['source_ref'], edge.provenance['source_version'], 1,
                            {'source': '0', 'target': '1', 'relation': 'supports', 'evidence': [edge.evidence]}, 'Old mismatch')
    with graph_for(project) as graph:
        graph.record_rejections([rejected])
    result = revalidate_proposals(project, apply=True)
    assert not result['accepted_claims'] and 'ranges were not recorded' in result['outcomes'][0]['reason']
