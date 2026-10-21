'''Persistent identities distinguish namesakes and retain supported aliases.'''

import json

import pytest

from osintgpt.graph import GraphStore, extract_document, neighbors, path_between, to_cypherl, to_json
from osintgpt.graph.identity import EntityRegistry
from osintgpt.graph.windowing import ExtractionOptions, Window, encode, units
from osintgpt.ingestion.documents import Document
from osintgpt.ingestion.provenance import text_spans
from osintgpt.ingestion.snapshots import TextSnapshot

from test_extraction_quality import Script, proposal


def extract(text, data, ref='one.md', known=()):
    return extract_document(Script(data), ref, text, known_entities=known,
                            options=ExtractionOptions(repair_quotes=False))


def test_namesakes_do_not_create_a_false_connection():
    first = extract('Alex hired Beta.', proposal('Alex hired Beta.', ('Alex', 'Beta')))
    second = extract('Alex hired Gamma.', proposal('Alex hired Gamma.', ('Alex', 'Gamma')),
                     ref='two.md', known=first.entities)
    assert first.entities[0].key != second.entities[0].key
    with GraphStore(':memory:') as graph:
        graph.add(first.entities + second.entities, first.edges + second.edges)
        assert path_between(graph, 'Beta', 'Gamma') is None
        with pytest.raises(ValueError, match='Ambiguous entity name; select an ID'):
            path_between(graph, 'Alex', 'Beta')
        with pytest.raises(ValueError, match='Ambiguous entity name; select an ID'):
            neighbors(graph, 'Alex')
        assert len(neighbors(graph, first.entities[0].key)) == 1
        assert path_between(graph, first.entities[0].key, 'Beta').length == 1
        exported = json.loads(to_json(graph))
        assert len({row['key'] for row in exported['entities'] if row['name'] == 'Alex'}) == 2
        assert all(edge['source_id'].startswith('ent_') for edge in exported['edges'])
        assert first.entities[0].key in to_cypherl(graph)


def test_same_name_in_one_response_stays_separate():
    text = 'Alex the doctor met Alex the lawyer.'
    data = {'entities': [{'id': 'doctor', 'name': 'Alex', 'evidence': ['Alex the doctor']},
                         {'id': 'lawyer', 'name': 'Alex', 'evidence': ['Alex the lawyer']}],
            'edges': [{'source': 'doctor', 'target': 'lawyer', 'relation': 'met', 'evidence': text}]}
    result = extract(text, data)
    assert len(result.entities) == 2
    assert result.edges[0].source_id != result.edges[0].target_id


def test_unsupported_reuse_keeps_a_separate_supported_mention():
    first = extract('Alex hired Beta.', proposal('Alex hired Beta.', ('Alex', 'Beta')))
    data = proposal('Alex hired Gamma.', ('Alex', 'Gamma'))
    data['entities'][0]['existing_id'] = first.entities[0].key
    result = extract('Alex hired Gamma.', data, 'two.md', first.entities)
    assert result.edges and result.entities[0].key != first.entities[0].key
    assert result.entities[0].provenance['resolution']['action'] == 'unresolved'
    assert any('Cross-document identity' in item.reason and item.status == 'warning' for item in result.rejected)


def test_unknown_existing_id_is_not_accepted():
    data = proposal('Alpha funds Beta.')
    data['entities'][0]['existing_id'] = 'ent_invented'
    result = extract('Alpha funds Beta.', data)
    assert result.edges and result.edges[0].source_id != 'ent_invented'
    assert any('not supplied' in item.reason and item.status == 'warning' for item in result.rejected)


def identified(name, number='123', issuer='Registro'):
    quote = f'{name}: {issuer} {number}.'
    return quote, {'id': 'subject', 'name': name, 'type': 'organization', 'evidence': [quote],
                   'identifiers': [{'issuer': issuer, 'value': number, 'evidence': [quote]}]}


def test_supported_unique_identifier_reuses_id_even_with_different_name():
    text, entity = identified('Alpha')
    first = extract(text, {'entities': [entity], 'edges': []})
    text, entity = identified('Alfa')
    second = extract(text, {'entities': [entity], 'edges': []}, 'two.md', first.entities)
    assert second.entities[0].key == first.entities[0].key
    assert 'Alfa' in second.entities[0].aliases
    assert len(second.entities[0].provenance['identity_sources']) == 2


@pytest.mark.parametrize('issuer,number', [('Registro', '124'), ('Otro', '123')])
def test_identifier_issuer_and_value_both_matter(issuer, number):
    text, entity = identified('Alpha')
    first = extract(text, {'entities': [entity], 'edges': []})
    text, entity = identified('Alpha', number, issuer)
    second = extract(text, {'entities': [entity], 'edges': []}, 'two.md', first.entities)
    assert second.entities[0].key != first.entities[0].key


def test_alias_needs_a_source_quote_and_is_searchable():
    text = 'Organización Alfa, también llamada OA, financia Beta.'
    data = proposal(text, ('Organización Alfa', 'Beta'))
    data['entities'][0]['aliases'] = [{'name': 'OA', 'evidence': [text]}]
    result = extract(text, data)
    with GraphStore(':memory:') as graph:
        graph.add(result.entities, result.edges)
        assert neighbors(graph, 'OA')[0].edge.source_id == result.entities[0].key
        assert next(entity for entity in graph.entities() if entity.aliases).provenance['alias_evidence']
    data['entities'][0]['aliases'][0]['name'] = 'Translated alias'
    warned = extract(text, data)
    assert warned.edges and not warned.entities[0].aliases
    assert warned.accounting.field_warnings == 1 and not warned.accounting.rejected_entities


def test_later_window_receives_bounded_ids_and_prior_source_context():
    text = 'Alpha funds Beta. ' + 'word ' * 100 + 'It still funds Beta.'

    class ContextModel(Script):
        def generate(self, system, user, *, max_output_tokens=None):
            payload = json.loads(user)
            if not self.calls:
                data = proposal('Alpha funds Beta.')
            elif len(self.calls) == 1:
                known = payload['known_entities']
                alpha = next(item for item in known if item['name'] == 'Alpha')
                assert 'Alpha' in alpha['source_context']['text']
                data = {'entities': [{'id': 'again', 'existing_id': alpha['id'], 'name': 'Alpha',
                                      'evidence': ['Alpha funds Beta.']}], 'edges': []}
            else:
                data = {'entities': [], 'edges': []}
            self.calls.append(payload)
            return json.dumps(data)

    generator = ContextModel()
    result = extract_document(generator, 'one.md', text, window_chars=180)
    assert result.ok and len(result.entities) == 2
    assert all(units(encode(call['known_entities'])) <= 4096 for call in generator.calls)


def test_context_candidate_limit_and_byte_budget():
    names = [f'Name{i:02}' for i in range(40)]
    text = ' '.join(names)
    data = {'entities': [{'id': name, 'name': name} for name in names], 'edges': []}
    result = extract(text, data)
    registry = EntityRegistry(result.entities)
    document = Document('one.md', text, spans=text_spans(text))
    version = result.entities[0].provenance['source_version']
    snapshot = TextSnapshot('one.md', 'hash', version, (document,))
    entries, _ = registry.context(document, snapshot, Window(0, len(text)), 100000)
    assert len(entries) == 32
    entries, _ = registry.context(document, snapshot, Window(0, len(text)), 1024)
    assert units(encode(entries)) <= 1024


def test_forgetting_source_removes_obsolete_ids_but_keeps_other_sources():
    first = extract('Alex hired Beta.', proposal('Alex hired Beta.', ('Alex', 'Beta')))
    second = extract('Alex hired Gamma.', proposal('Alex hired Gamma.', ('Alex', 'Gamma')),
                     ref='two.md', known=first.entities)
    with GraphStore(':memory:') as graph:
        graph.add(first.entities + second.entities, first.edges + second.edges)
        graph.forget(['one.md'])
        assert {entity.key for entity in graph.entities()} == {entity.key for entity in second.entities}
        assert path_between(graph, 'Alex', 'Gamma').length == 1
