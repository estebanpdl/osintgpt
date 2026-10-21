'''Materialize annotated text, structured records and cached scanned pages in isolated projects.'''

import csv
import json
from pathlib import Path

from osintgpt import Project
from osintgpt.ingestion import Corpus, FieldMapping
from osintgpt.ingestion.pdf import render_page
from osintgpt.ingestion.transcription import cache_path
from osintgpt.llm.usage import Usage, UsageRecorder


def prepare_case(root, case):
    project = Project.create(case['id'], home=root).with_settings(graph_enabled=True)
    ref = case['ref'].split('#', 1)[0]
    path = project.paths.root / ref
    mapping = None
    if case['storage'] in ('csv', 'json'):
        rows = [{'id': 'r1', 'text': case['text']}]
        if case['storage'] == 'json':
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf8')
        else:
            with path.open('w', encoding='utf8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'text'])
                writer.writeheader()
                writer.writerows(rows)
        mapping = FieldMapping(content=('text',), identity='id')
    elif case['storage'] == 'scan':
        from PIL import Image, ImageDraw
        picture = Image.new('RGB', (800, 200), 'white')
        ImageDraw.Draw(picture).text((20, 60), case['text'], fill='black')
        picture.save(path, 'PDF')
        cache_path(project.paths.extracts, render_page(path, 0)).write_text(case['text'], encoding='utf8')
    else:
        path.write_text(case['text'], encoding='utf8')
    Corpus.load(project.paths.sources).register(ref, mapping=mapping)
    return project


def gold_proposal(case):
    entities = [{'id': entity['id'], 'name': entity['names'][0], 'type': '', 'description': '',
                 'existing_id': '', 'identifiers': [], 'evidence': entity['anchors'],
                 'aliases': [{'name': name, 'evidence': entity['anchors']} for name in entity['names'][1:]]}
                for entity in case['entities']]
    edges = [{'source': edge['source'], 'target': edge['target'], 'relation': edge['relations'][0],
              'evidence': edge['evidence'], **edge['qualifiers']} for edge in case['edges']]
    return {'entities': entities, 'edges': edges}


class AnnotationReplay:
    provider = 'annotation-replay'
    model = 'annotation-replay'

    def __init__(self, case):
        self.case = case
        self.recorder = UsageRecorder()

    def generate(self, system, user, *, max_output_tokens=None):
        self.recorder.record(Usage(self.provider, self.model, billable=False, counted=False))
        return json.dumps(gold_proposal(self.case), ensure_ascii=False)


def write_report(path, report):
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
