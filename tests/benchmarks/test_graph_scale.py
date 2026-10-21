'''The native transport fixture exercises recovery without network traffic.'''

from graph_scale import run


def test_offline_pdf_recovery_and_replacement():
    result = run(pages=30)
    assert result['passed'] and result['actual_provider_spend_usd'] == 0
    stages = {stage['name']: stage for stage in result['stages']}
    assert stages['rate_limit_then_interrupt']['counts']['retries'] == 1
    assert stages['resume_to_completion']['counts']['responses_reused'] == 1
    assert stages['unchanged_reuse']['transport_attempts'] == 0
    assert stages['changed_source_failed_replacement']['published_edges'] == 30
    assert stages['changed_source_recovery']['published_edges'] == 30
