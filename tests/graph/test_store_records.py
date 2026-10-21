'''SQLite record behavior, independent of model extraction.'''

from osintgpt.graph import Entity, merge_key
from test_graph import edge, store


class TestStore:
    def test_an_identical_claim_is_stored_once(self, store):
        store.add([], [edge('A', 'B')])
        store.add([], [edge('A', 'B')])

        assert store.edge_count == 1

    def test_the_same_claim_from_two_documents_is_two_claims(self, store):
        '''Independent sources must remain separately inspectable.'''
        store.add([], [edge('A', 'B', ref='one.md'),
                       edge('A', 'B', ref='two.md')])

        assert store.edge_count == 2

    def test_an_entity_keeps_its_first_spelling(self, store):
        store.add([Entity(key=merge_key('Alpha Corp'), name='Alpha Corp',
                          type='org', mentions=1)], [])
        store.add([Entity(key=merge_key('alpha corp'), name='alpha corp',
                          type='', mentions=1)], [])

        entities = store.entities()

        assert len(entities) == 1
        assert entities[0].name == 'Alpha Corp'
        assert entities[0].mentions == 2

    def test_forgetting_a_document_drops_its_claims_only(self, store):
        store.add([], [edge('A', 'B', ref='one.md'),
                       edge('C', 'D', ref='two.md')])

        assert store.forget(['one.md']) == 1
        assert [e.ref for e in store.edges()] == ['two.md']

    def test_an_empty_graph_is_not_built(self, store):
        assert store.is_built is False

    def test_it_becomes_built_once_something_is_stored(self, store):
        store.add([], [edge('A', 'B')])

        assert store.is_built is True


class TestEndpointNames:
    def test_an_endpoint_keeps_the_form_the_document_used(self, store):
        '''Manual edge insertion must preserve the original endpoint spelling.'''
        store.add([], [edge('Alpha Corp', 'Beta Ltd')])

        stored = store.edges()[0]

        assert stored.source == 'Alpha Corp'
        assert stored.target == 'Beta Ltd'

    def test_an_endpoint_does_not_inflate_mention_counts(self, store):
        '''An edge alone does not add an extracted entity mention.'''
        store.add([], [edge('Alpha', 'Beta')])

        assert {e.name: e.mentions for e in store.entities()} == {
            'Alpha': 0, 'Beta': 0
        }

    def test_an_extracted_entity_keeps_its_own_count(self, store):
        store.add(
            [Entity(key=merge_key('Alpha'), name='Alpha', type='org',
                    mentions=5)],
            [edge('Alpha', 'Beta')]
        )

        counts = {e.name: e.mentions for e in store.entities()}

        assert counts['Alpha'] == 5

    def test_an_endpoint_does_not_overwrite_a_known_type(self, store):
        store.add(
            [Entity(key=merge_key('Alpha'), name='Alpha', type='organization',
                    mentions=1)],
            []
        )
        store.add([], [edge('Alpha', 'Beta')])

        types = {e.name: e.type for e in store.entities()}

        assert types['Alpha'] == 'organization'

    def test_endpoints_in_another_script_keep_their_form(self, store):
        store.add([], [edge('Альфа', 'Бета')])

        stored = store.edges()[0]

        assert (stored.source, stored.target) == ('Альфа', 'Бета')
