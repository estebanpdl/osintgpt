'''Evidence quality survives persistence, model retrieval, export, and rebuilds.'''

import json

import pytest

from osintgpt import Project
from osintgpt.agentic import ToolContext, graph_query
from osintgpt.graph import ExtractionOptions, build_graph, graph_for, saved_graph_status, to_cypherl, to_json, verify_evidence
from osintgpt.ingestion import Corpus
from osintgpt.llm.usage import CostLimitReached, Usage, UsageRecorder
from test_extraction_quality import Script, proposal


@pytest.fixture
def project(tmp_path):
    project = Project.create('Quality', path=tmp_path / 'project').with_settings(graph_enabled=True)
    project.save()
    (project.paths.root / 'a.md').write_text('Alpha did not fund Beta. The claim dates to 2026.')
    Corpus.load(project.paths.sources).register('a.md')
    return project


def test_audit_survives_reopening_and_incremental_skip_keeps_gaps(project):
    report = build_graph(project, Script(proposal('Invented.')),
                         options=ExtractionOptions(repair_quotes=False))
    assert report.status == 'complete_with_gaps' and report.edges == 0
    with graph_for(project) as graph:
        audit = graph.rejections(['a.md'])
        assert len(audit) == 1 and audit[0].source_version and audit[0].window == 1
    second = build_graph(project, Script({'entities': [], 'edges': []}), incremental=True,
                         options=ExtractionOptions(repair_quotes=False))
    assert second.calls == 0 and second.status == 'complete_with_gaps'
    fixed = build_graph(project, Script(proposal('Alpha did not fund Beta.', relation_polarity='negative')), rebuild=True)
    assert fixed.status == 'complete' and fixed.edges == 1
    with graph_for(project) as graph:
        assert len(graph.rejections()) == 1
        assert len(graph.entities()) == 2


def test_model_and_exports_receive_all_qualifications_and_excerpts(project):
    quotes = ['Alpha did not fund Beta.', 'The claim dates to 2026.']
    data = proposal(quotes, relation_polarity='negative', assertion='alleged', statement_time='2026')
    assert build_graph(project, Script(data)).status == 'complete'
    tool = graph_query(ToolContext(project, None), entity='Alpha')
    claim = tool.payload['claims'][0]
    assert (claim['relation_polarity'], claim['assertion'], claim['statement_time']) == ('negative', 'alleged', '2026')
    assert claim['evidence_quotes'] == quotes and claim['source_id'].startswith('ent_')
    with graph_for(project) as graph:
        exported = json.loads(to_json(graph))['edges'][0]
        assert exported['relation_polarity'] == claim['relation_polarity'] and exported['evidence_quotes'] == quotes
        cypher = to_cypherl(graph)
        assert 'relation_polarity: "negative"' in cypher and 'assertion: "alleged"' in cypher
    assert verify_evidence(project).found == 1
    (project.paths.root / 'a.md').write_text(quotes[0])
    verified = verify_evidence(project).results[0]
    assert verified.status == 'not_found' and verified.snapshot_status == 'found'


def test_repairs_are_included_in_usage_and_durable_audit(project):
    class Metered(Script):
        recorder = UsageRecorder()

        def generate(self, system, user, *, max_output_tokens=None):
            self.recorder.record(Usage('stub', 'test', 10, 5, billable=False))
            return super().generate(system, user, max_output_tokens=max_output_tokens)

    generator = Metered(proposal('Alpha did not funded Beta.', relation_polarity='negative'),
                        {'corrections': [{'index': 0, 'evidence': ['Alpha did not fund Beta.']}]})
    report = build_graph(project, generator)
    assert report.calls == report.usage['calls'] == 2
    assert report.usage['total_tokens'] == 30 and report.usage['complete']
    assert report.accounting.repaired_claims == 1
    with graph_for(project) as graph:
        item = graph.rejections()[0]
        assert item.status == 'repaired' and item.corrected_evidence == ('Alpha did not fund Beta.',)


def test_cost_ceiling_interrupts_correction_after_accounting_response(project):
    class Metered(Script):
        recorder = UsageRecorder(cost_ceiling_usd=1)

        def generate(self, system, user, *, max_output_tokens=None):
            self.recorder.record(Usage('stub', 'unpriced', 10, 5, billable=bool(self.calls)))
            return super().generate(system, user, max_output_tokens=max_output_tokens)

    with pytest.raises(CostLimitReached):
        build_graph(project, Metered(proposal('Invented.')))
    status = saved_graph_status(project)
    assert status['status'] == 'interrupted'
    assert status['counts']['requests_attempted'] == status['counts']['requests_succeeded'] == 2
    assert status['usage']['calls'] == 2 and status['usage']['total_tokens'] == 30
