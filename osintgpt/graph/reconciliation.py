'''Plan source refreshes without mistaking unavailable registered files for removals.'''

from osintgpt.ingestion import Corpus, is_image
from osintgpt.ingestion.snapshots import source_fingerprint, transcriptions_current
from osintgpt.projects.paths import CANON_DIR

from .recipes import changed_fields, compatible_recipe
from .model_metadata import model_identity, model_label, recipe_data as saved_recipe
from .state import source_ref


def registered(corpus, root, ref):
    path = (root / ref).resolve()
    roots = [(root / source.path).resolve() for source in corpus]
    roots.append((root / CANON_DIR).resolve())
    return any(path == parent or path.is_relative_to(parent) for parent in roots)


def inventory(project, graph):
    root = project.paths.root
    if not project.paths.sources.is_file() and graph.source_refs():
        raise ValueError('Source registrations are unavailable; existing graph contributions were preserved.')
    corpus = Corpus.load(project.paths.sources)
    files = {source_ref(path, root): path for path in corpus.files(root)}
    missing, removed = [], []
    for ref in sorted(graph.source_refs() - files.keys()):
        (missing if registered(corpus, root, ref) else removed).append(ref)
    for source in corpus:
        path = root / source.path
        if not path.exists():
            ref = source_ref(path, root)
            covered = any((root / item).resolve().is_relative_to(path.resolve()) for item in missing)
            if ref not in missing and not covered:
                missing.append(ref)
    images = sum(is_image(path) for path in files.values())
    return corpus, {ref: path for ref, path in files.items() if not is_image(path)}, missing, removed, images


def refresh_reason(previous, snapshot, recipe, recipe_data):
    if not previous:
        return 'New source'
    changes = refresh_changes(previous, snapshot, recipe_data)
    reasons = ['Source text or field mapping changed'] if changes['source_changed'] else []
    if previous['recipe'] != recipe and not compatible_recipe(saved_recipe(previous), recipe_data):
        reasons.append('Extraction settings changed: ' + changed_fields(saved_recipe(previous), recipe_data))
    if changes['model_changed']:
        reasons.append(f"Graph model: {model_label(changes['previous_model'])} -> {model_label(changes['next_model'])}")
    return '; '.join(reasons)


def refresh_changes(previous, snapshot, recipe):
    old = saved_recipe(previous)
    before, after = model_identity(old), model_identity(recipe)
    return {'source_changed': bool(previous and previous['source_version'] != snapshot.version),
            'extraction_changes': [key for key in recipe if previous and old.get(key) != recipe[key]],
            'model_changed': bool(previous and before != after),
            'previous_model': before, 'next_model': after}


def validate_current(project, path, snapshot):
    corpus = Corpus.load(project.paths.sources)
    if not registered(corpus, project.paths.root, snapshot.ref):
        raise ValueError('Source was unregistered during extraction; rerun to reconcile the graph.')
    mapping = corpus.mapping_for(path, project.paths.root)
    if (source_fingerprint(path, mapping) != snapshot.fingerprint
            or not transcriptions_current(snapshot, project.paths.extracts)):
        raise ValueError('Source changed during graph extraction; previous published claims were preserved.')


def refresh_label(reason):
    '''Operator summary; exact fingerprint fields remain available as diagnostics.'''
    prefix = 'Extraction settings changed: '
    if prefix not in reason:
        return reason.replace('; saved windows discarded', '; earlier checkpoints cannot be reused')
    before, fields = reason.split(prefix, 1)
    fields = fields.split(';', 1)[0].split(', ')
    labels = {'provider': 'graph model', 'evidence_policy': 'evidence matching',
              'identity_policy': 'identity resolution', 'identity_tools': 'entity lookup',
              'extraction_prompt': 'extraction instructions', 'repair_prompt': 'correction instructions',
              'response_schema': 'response structure', 'output_policy': 'provider output policy',
              'options': 'request context settings', 'windowing_version': 'source window planning',
              'window_chars': 'source window planning', 'overlap': 'source window planning',
              'version': 'extraction policy', 'text_version': 'text extraction'}
    return before + 'Changed ' + ', '.join(dict.fromkeys(labels.get(field, 'extraction settings') for field in fields)) + '.'
