'''Render qualified assertions and source context without contacting a model.'''

from osintgpt.graph import assertion_context
from osintgpt.graph.inspection import inspect_assertion
from osintgpt.graph.records import normalize_relation_polarity


def render_claim(st, row):
    row = normalize_relation_polarity(row)
    st.text(f"{row['source']} — {row['relation']} → {row['target']}")
    st.caption(f"{row.get('assertion_id', 'Assertion ID not recorded')} · {row['ref']}")
    qualifications = [f"{name.replace('_', ' ').capitalize()}: {row.get(name) or 'unknown'}" for name in
                      ('relation_polarity', 'assertion', 'attributed_to', 'event_time', 'statement_time')]
    st.caption(' · '.join(qualifications))
    st.caption('Relation polarity: positive, negative, or unknown. '
               'Assertion status separately records allegations or uncertainty. Dates retain source wording.')
    if 'quotation_preview' in row:
        st.text('Quotation preview: ' + row['quotation_preview'])
        st.caption(f"{row['quotation_count']} supporting quotations. Select this assertion below for complete evidence.")
        if row.get('traversal', {}).get('against_direction'):
            st.caption('This step was traversed against the stored arrow; the relationship above retains its direction.')
        return
    for index, quote in enumerate(row.get('evidence_quotes') or [row.get('evidence', '')]):
        st.text(f'Quotation {index + 1}: {quote}')
    provenance = row.get('provenance', {})
    supporting = provenance.get('supporting', [])
    matches = [match for item in supporting for match in item.get('matches', [])] or provenance.get('matches', [])
    spans = [span for match in matches for span in match.get('spans', [])]
    labels = list(dict.fromkeys(f"page {span['page']}" for span in spans if span.get('page')))
    labels += list(dict.fromkeys(f"section {span['section']}" for span in spans if span.get('section')))
    version = row.get('source_version') or provenance.get('source_version', '')
    st.caption('Saved text version: ' + (version or 'unversioned') + (' · ' + '; '.join(labels) if labels else ''))
    if matches:
        st.caption('Quotation offsets in extracted record text: ' + ', '.join(f"{m['start']}–{m['end']}" for m in matches))
        st.caption('Evidence match: ' + ', '.join(sorted({m.get('method', 'previous matcher') for m in matches})))
        for match in matches:
            if match.get('method') == 'formatting':
                st.text('Original source excerpt: ' + match['source_excerpt'])
    decisions = provenance.get('identity_decisions', {})
    if any(item.get('action') == 'unresolved' for item in decisions.values()):
        st.info('An endpoint has unresolved identity. This assertion supports its separate mention, not equivalence to another node.')
    if decisions:
        with st.expander('Endpoint identity decisions'):
            st.json(decisions)
    if row.get('verification'):
        check = row['verification']
        st.caption(f"Saved evidence: {check.get('snapshot', 'unknown')}; current evidence: {check.get('current', 'unknown')}; "
                   f"source state: {check.get('source_state', 'unknown')}. Textual support does not establish truth.")
    step = row.get('traversal', {})
    if step.get('against_direction'):
        st.caption('This step was traversed against the stored arrow. The assertion above keeps its original meaning.')


def render(st, project, assertion_id, state, *, inspection=None):
    result = (inspection.inspect_assertion(assertion_id) if inspection is not None
              else inspect_assertion(project, assertion_id))
    rows = result.get('claims', [])
    if not rows:
        st.warning(result.get('note', 'No complete assertion could be read. Try a narrower query.'))
        return
    selected = rows[0]
    st.markdown('#### Selected assertion')
    render_claim(st, selected)
    if result['truncated']:
        st.info('Additional assertions were limited: ' + ', '.join(result['truncation_reasons']))
    prefix = f'graph-evidence:{project.id}:{assertion_id}:'
    quotes = selected['evidence_quotes']
    quote = st.selectbox('Supporting quotation', list(range(len(quotes))), format_func=lambda index: f'Quotation {index + 1}', key=prefix + 'quote')
    support = selected.get('provenance', {}).get('supporting', [])
    matches = support[quote].get('matches', []) if quote < len(support) else []
    occurrence = st.number_input('Quotation occurrence (zero-based)', min_value=0, max_value=max(0, len(matches) - 1),
                                 value=0, step=1, key=prefix + f'occurrence:{quote}')
    offset_key = prefix + f'offset:{quote}:{occurrence}'
    offset = state.get(offset_key, 0)
    if st.button('Read source context', key=prefix + 'read'):
        state[offset_key] = 0
        state[prefix + 'reading'] = (quote, occurrence)
        offset = 0
    if state.get(prefix + 'reading') == (quote, occurrence):
        context = assertion_context(project, assertion_id, quote_index=quote, occurrence=occurrence,
                                    offset=offset, allow_disabled=True)
        if context['status'] != 'ok':
            st.warning(context.get('problem', context.get('note', 'Source context unavailable.')))
        else:
            st.text(context['text'])
            st.caption(f"Saved source context, characters {context['start']}–{context['end']}; "
                       f"quotation complete in this page: {'yes' if context['quote_complete'] else 'no'}.")
            if 'next_offset' in context and st.button('Next context page', key=prefix + 'next'):
                state[offset_key] = context['next_offset']
                st.rerun()
    if len(rows) > 1:
        with st.expander(f'Other assertions for this entity pair ({len(rows) - 1})'):
            st.caption('Compare quotations, relation labels, attribution, and dates. Different relation polarity may reflect '
                       'conflicting accounts or different circumstances; it is not automatically a contradiction.')
            for row in rows[1:]:
                render_claim(st, row)
