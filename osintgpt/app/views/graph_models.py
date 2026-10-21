'''Shared graph model selection and saved-publication displays for app views.'''

from osintgpt.graph.model_metadata import model_label, selection_label
from osintgpt.graph.provider import resolve_graph_model
from osintgpt.llm.registry import GENERATION_BACKENDS


def controls(st, project, defaults, config, changes, model_help):
    effective = project.effective_settings(defaults)
    prefix = f'graph-model:{project.id}:'
    st.markdown('#### Graph extraction')
    inherit = st.checkbox('Use Chat/RAG settings', value=effective.graph_model_mode == 'inherit', key=prefix + 'inherit',
        help='Uses the Chat/RAG provider and model for extraction and quotation correction. '
             'Changing Chat/RAG then affects the next explicit graph build. This choice overrides user graph defaults.')
    choices = {'graph_model_mode': 'inherit' if inherit else 'independent'}
    if not inherit:
        providers = sorted(GENERATION_BACKENDS)
        saved_provider = effective.graph_generation_provider or project.settings.graph_generation_provider
        if saved_provider and saved_provider not in providers:
            providers.append(saved_provider)
        provider = st.selectbox('Graph extraction provider', providers,
                               index=providers.index(saved_provider) if saved_provider in providers else 0,
                               key=prefix + 'provider')
        saved_model = (effective.graph_generation_model or project.settings.graph_generation_model
                       if provider == saved_provider else '')
        model = st.text_input('Graph extraction model', saved_model, key=prefix + f'model:{provider}',
                             help=model_help('generation', provider, GENERATION_BACKENDS)).strip()
        choices.update(graph_generation_provider=provider, graph_generation_model=model)
    proposed = project.with_settings(**{**changes, **choices})
    try:
        selected = resolve_graph_model(proposed.effective_settings(defaults), proposed.settings_for(config, defaults))
    except ValueError as error:
        st.warning(f'Graph model selection needs attention: {error}')
        selected = None
    if selected:
        st.caption('Graph extraction selection: ' + selection_label({**selected.identity(), 'mode': selected.mode}))
        field = GENERATION_BACKENDS[selected.provider].settings_field
        available = not field or bool(getattr(config, field, None))
        st.caption(f'Graph provider credential: {"not required" if not field else "available" if available else "not set"}. '
                   'Availability is checked locally; provider access has not been tested.')
    st.caption('Save to apply these choices to CLI and app builds. Saving makes no model requests. '
               'Preview the workload in Graph before building with a changed extraction model.')
    # Inherited projects with no configured chat model must still allow other settings to be saved.
    return choices, inherit or selected is not None


def saved_models(st, status):
    models = status.get('extraction_models', {})
    rows = models.get('sources', [])
    if rows or models.get('unattributed_entities'):
        label = 'multiple models' if models.get('mixed') else 'one recorded model' if any(row['known'] for row in rows) else 'unknown model'
        st.caption(f'Published graph: {label}; {models.get("unknown_sources", 0)} sources with unknown extraction model. '
                   'This describes saved publications, not the next-build setting.')
        with st.expander('Published extraction models'):
            st.dataframe([{'Source': row['ref'], 'Source version': row['source_version'],
                           'Extraction model': model_label(row['identity']), 'Identity known': row['known']}
                          for row in rows], hide_index=True)
            if models.get('unattributed_entities'):
                st.caption(f"{models['unattributed_entities']} entities have no source-level model attribution.")
    run_model = status.get('run_model')
    if run_model:
        st.caption('Latest run selection: ' + selection_label(run_model) + '. Reused sources retain their saved model attribution.')
