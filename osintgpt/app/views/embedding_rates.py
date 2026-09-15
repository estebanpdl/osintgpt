'''Settings controls for the selected paid embedding provider's quotas.'''

from osintgpt.rate_settings import PAID_EMBEDDING_PROVIDERS


def controls(st, effective, provider):
    if provider not in PAID_EMBEDDING_PROVIDERS:
        return {}
    st.markdown('#### Embedding rate limits')
    st.caption(
        f'These limits apply to {provider}. Enter the quota shown in your '
        'provider dashboard, or a lower budget for this workload. Blank inherits '
        'user defaults; 0 removes that operator ceiling.'
    )
    changes = {}
    for suffix, label, help_text in (
        ('rpm', 'Embedding requests per minute (RPM)', 'Spaces requests and limits their count over a rolling minute.'),
        ('tpm', 'Embedding tokens per minute (TPM)', 'Limits input tokens over a rolling minute. Higher tiers can use higher values.'),
        ('batch_tokens', 'Embedding tokens per request', 'Optional smaller batch target. Provider request ceilings still apply.')
    ):
        name = f'{provider}_embedding_{suffix}'
        changes[name] = st.number_input(
            label, min_value=0, value=getattr(effective, name), step=1,
            key=name, help=help_text
        )
    name = f'{provider}_embedding_quota_scope'
    changes[name] = st.text_input(
        'Shared embedding quota label', getattr(effective, name), key=name,
        help='Use the same label across keys/projects/models sharing one provider quota. '
             'Blank groups by API key and model. Coordination covers runs using the same osintgpt home.'
    ).strip()
    if provider == 'openai':
        st.caption('OpenAI response headers can supply limits after the first request. A configured lower limit still applies.')
    else:
        st.caption('Set RPM and TPM explicitly for this provider. Token pacing uses a conservative local estimate, not billing usage.')
    return changes
