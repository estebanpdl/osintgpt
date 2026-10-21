'''Identify extraction work without storing credentials or usage limits.'''

import hashlib
import json
from dataclasses import asdict

from osintgpt.ingestion.snapshots import EXTRACTION_VERSION
from osintgpt.prompts import prompt

from .contract import EXTRACTION_SCHEMA, REPAIR_SCHEMA
from .windowing import ExtractionOptions, WINDOWING_VERSION

RECIPE_VERSION = 5

# Only these reviewed naming-only changes are equivalent. Actual request hashes remain exact.
RELATION_FIELD_RENAME = {
    'response_schema': ('d01ae4320d113550f0415ee06949fe841e25f607c311bdcae8fd74f548835839',
                        '6f807e844cb4f73d6d718f250a4031cf2dd0baf4cd61299153bb8f3f98ad6c49'),
    'extraction_prompt': ('66f97cfa4aba867936e5a5a1831853d8d76388d1366eed9237ae04d56045b01b',
                          'a5e5cfa1dd5f24d01d3cb71569e82d92ae6b41bd1f4ac34ec9da311b55d4eee2'),
    'repair_prompt': ('baef3d190403d661c69972b2b9d1fa6abedc6bb22ca47b719ca8010fbde09e78',
                      '4d009f790c3930206b23a130b8a46c65dd50089d84f8b662b4cb0e54f4698267'),
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def extraction_recipe(generator, options=None):
    from .extraction import WINDOW_CHARS, WINDOW_OVERLAP
    from .output_policy import identity, policy_data
    from .identity_tools import RESOLVING_SCHEMA, TOOLS
    budgets = asdict(options or ExtractionOptions())
    budgets.pop('model_input_units')
    return {'version': RECIPE_VERSION, 'text_version': EXTRACTION_VERSION,
            'provider': identity(generator), 'options': budgets, 'output_policy': policy_data(generator),
            'window_chars': WINDOW_CHARS, 'overlap': WINDOW_OVERLAP,
            'windowing_version': WINDOWING_VERSION,
            'response_schema': fingerprint([EXTRACTION_SCHEMA, RESOLVING_SCHEMA, REPAIR_SCHEMA]),
            'evidence_policy': 2, 'identity_policy': 1, 'identity_tools': fingerprint(TOOLS),
            'extraction_prompt': fingerprint(prompt('graph_extraction')),
            'repair_prompt': fingerprint(prompt('graph_quote_repair'))}


def changed_fields(previous, current):
    return ', '.join(key for key in current if previous.get(key) != current[key])


def prior_relation_field(recipe):
    """Recognize the reviewed pre-rename instructions in a retained compatible job."""
    return bool(recipe and recipe.get('version') in (4, 5) and
                all(recipe.get(key) == values[0] for key, values in RELATION_FIELD_RENAME.items()))


def compatible_recipe(previous, current):
    '''Same extraction semantics; legacy source plans remain pinned in their jobs.'''
    if isinstance(previous, str):
        previous = json.loads(previous)
    if previous.get('version') not in (4, 5) or current.get('version') not in (4, 5):
        return previous == current
    def semantic(data):
        options = data.get('options', {})
        renamed = {key: RELATION_FIELD_RENAME[key][1] if key in RELATION_FIELD_RENAME
                   and value == RELATION_FIELD_RENAME[key][0] else value for key, value in data.items()}
        return {**{k: v for k, v in renamed.items() if k not in {'version', 'options'}},
                'options': {k: options.get(k) for k in ('context_budget', 'repair_quotes')}}
    return semantic(previous) == semantic(current)
