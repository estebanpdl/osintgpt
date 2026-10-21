'''Validate complete JSON and correct a failed contract without re-extracting a document.'''

from .calling import request_schema
from .checkpoints import SavedInvalidResponse
from .grounding import parse_response
from .windowing import encode


def structured_generate(calls, system, payload, schema, *, purpose='extraction', key='edges'):
    try:
        reply = calls.generate(system, encode(payload), schema, purpose=purpose)
    except SavedInvalidResponse as error:
        # Refusals/truncation are not JSON repair candidates.
        error.response.require_complete()
        reply = error.response.text
    return validate_completion(calls, reply, system, payload, schema, key=key)


def validate_completion(calls, reply, system, payload, schema, *, key='edges'):
    try:
        data = parse_response(reply, key, schema=request_schema(calls.generator, schema))
        calls.validated()
        return data
    except ValueError as error:
        calls.invalidate(str(error))
        calls.counts.invalid_responses += 1
        problem = str(error)
    purpose = calls.purpose + ':structured-correction'
    correction = {**payload, 'completion_problem': problem,
                  'instruction': 'Return the complete requested JSON structure, grounded only in the original source. '
                                 'The previous response is retained for diagnosis. Do not invent missing evidence.'}
    with_previous = {**correction, 'previous_response': reply}
    if calls.fits(system, encode(with_previous), schema):
        correction = with_previous
    calls.emit('correcting_structure')
    calls.counts.structured_corrections += 1
    try:
        repaired = calls.generate(system, encode(correction), schema, purpose=purpose, stage='structured_correction')
        data = parse_response(repaired, key, schema=request_schema(calls.generator, schema))
    except ValueError as error:
        calls.invalidate(str(error))
        calls.counts.invalid_responses += 1
        raise ValueError('Structured completion made no progress; original and correction responses are saved. ' + str(error)) from error
    calls.validated()
    return data
