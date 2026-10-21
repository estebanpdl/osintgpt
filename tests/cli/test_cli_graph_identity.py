'''Operator identity actions use shared local services and require explicit apply.'''

import json

from osintgpt.graph import Entity, Edge, graph_for, saved_graph_status
from test_cli_graph import clear_credentials, create_project, home, invoke, runner


def test_identity_review_preview_apply_and_revalidation_commands(runner, home, monkeypatch, cli_help):
    project = create_project(runner, home)
    with graph_for(project) as graph:
        graph.add([Entity('one', 'Orion'), Entity('two', 'Orion'), Entity('lyra', 'Lyra'), Entity('vega', 'Vega')],
                  [Edge('Orion', 'Lyra', 'supports', 'one.md', 'Orion supports Lyra.', source_id='one', target_id='lyra'),
                   Edge('Orion', 'Vega', 'supports', 'two.md', 'Orion supports Vega.', source_id='two', target_id='vega')])
    clear_credentials(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError('Identity review must not construct a provider.')
    monkeypatch.setattr('osintgpt.llm.build_generation_provider', forbidden)
    result = invoke(runner, home, 'graph', 'reconcile', '--query', 'Orion', '--project', project.slug, '--json')
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)['candidates']) == 2
    args = ('graph', 'reconcile', '--from-id', 'one', '--to-id', 'two',
            '--reason', 'Reviewed matching source context.', '--project', project.slug, '--json')
    preview = invoke(runner, home, *args)
    assert preview.exit_code == 0 and json.loads(preview.stdout)['status'] == 'preview'
    assert saved_graph_status(project)['entities'] == 4
    applied = invoke(runner, home, *args, '--apply')
    assert applied.exit_code == 0 and json.loads(applied.stdout)['backup']
    assert saved_graph_status(project)['entities'] == 3
    local = invoke(runner, home, 'graph', 'revalidate', '--project', project.slug, '--json')
    assert local.exit_code == 0 and json.loads(local.stdout)['provider_calls'] == 0
    assert '--split-source' in cli_help('graph', 'reconcile')
    assert '--apply' in cli_help('graph', 'revalidate')
