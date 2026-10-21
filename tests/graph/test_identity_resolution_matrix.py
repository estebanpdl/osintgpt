'''Reviewed fictional proposals measure merges and abstentions, not live model quality.'''

import pytest

from osintgpt.graph import extract_document
from osintgpt.graph.grounding import EvidenceScope
from osintgpt.graph.identity import EntityRegistry
from osintgpt.graph.identity_tools import IdentityTools
from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import text_spans
from osintgpt.ingestion.snapshots import TextSnapshot
from test_extraction_quality import Script, proposal


@pytest.mark.parametrize('name,quote', [
    ('Instituto Orión', 'Instituto Orión, fundado por Mira, desarrolla Prisma.'),
    ('مؤسسة نور', 'مؤسسة نور، أسستها ميرا، تطور مشروع بريزما.'),
    ('星河研究所', '星河研究所由米拉创立，研发棱镜项目。'),
])
@pytest.mark.parametrize('name_only', [False, True])
def test_contextual_identity_matrix_retains_mentions_without_name_only_merges(name, quote, name_only):
    first = extract_document(Script({'entities': [{'id': 'old', 'name': name, 'type': 'organization',
        'evidence': [quote]}], 'edges': []}), 'prior.md', quote).entities[0]
    registry = EntityRegistry([first])
    lookup = IdentityTools(registry, 4096)
    lookup.run('find_entity_candidates', {'queries': [name]})
    lookup.run('read_entity_context', {'entity_id': first.key})
    registry.candidate_evidence = lookup.evidence
    doc = Document('current.md', quote, spans=text_spans(quote))
    scope = EvidenceScope(doc, TextSnapshot('current.md', 'hash', 'version', (doc,)), [(0, len(quote))])
    row = {'id': 'new', 'name': name, 'type': 'organization', 'existing_id': first.key, 'evidence': [quote],
           'resolution': {'action': 'reuse', 'method': 'context', 'reason': 'Same named founder and named project.',
                          'type_compatible': True, 'type_reason': 'Both explicitly describe an organization.',
                          'evidence': [name if name_only else quote], 'candidate_evidence': [quote]}}
    entities, _ = registry.resolve([row], scope, lookup.allowed)
    entity = next(iter(entities.values()))
    assert (entity.key == first.key) is not name_only
    assert entity.provenance['resolution']['action'] == ('unresolved' if name_only else 'reuse')
    assert len(entity.provenance['mentions']) == (1 if name_only else 2)


def test_entity_quote_correction_precedes_dependent_relationship_validation():
    text = 'Orion supports Lyra.'
    data = proposal(text, ('Orion', 'Lyra'))
    data['entities'][0]['evidence'] = ['Orion has invented attributes.']
    model = Script(data, {'corrections': [{'index': 0, 'evidence': [text]}]})
    result = extract_document(model, 'fiction.md', text)
    assert result.ok and len(result.entities) == 2 and len(result.edges) == 1
    assert len(model.calls) == 2 and result.accounting.repaired_entities == 1
    assert result.accounting.rejected_entities == result.accounting.rejected_claims == 0
    assert result.rejected[0].status == 'repaired' and result.rejected[0].candidate['kind'] == 'entity'
