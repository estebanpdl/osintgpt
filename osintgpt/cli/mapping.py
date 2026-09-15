'''Field roles as typed at a prompt, shared by the commands that accept them.'''

from typing import Dict, List, Optional

from osintgpt.ingestion import FieldMapping

LIST_KEYS = {'content', 'metadata'}
SINGLE_KEYS = {'timestamp', 'author', 'identity', 'records'}
COUNT_KEYS = {'min_chars'}

# Shown when a structured file arrives without a mapping. Every role is named,
# not only content: the rest are what make a corpus filterable and citable,
# and nothing later can recover what was never separated out.
MAPPING_EXAMPLE = (
    '--map content=<field> --map timestamp=<field> --map author=<field> '
    '--map identity=<field> --map metadata=<field>,<field> --map min_chars=40'
)


def build_mapping(pairs: Optional[List[str]]) -> FieldMapping:
    roles: Dict[str, object] = {}
    for pair in pairs or []:
        key, separator, value = pair.partition('=')
        key = key.strip()
        if not separator or not value.strip():
            raise ValueError(f'--map {pair!r} should look like key=value')

        if key in LIST_KEYS:
            values = tuple(
                item.strip() for item in value.split(',') if item.strip()
            )
            roles[key] = tuple(roles.get(key, ())) + values
        elif key in SINGLE_KEYS:
            roles[key] = value.strip()
        elif key in COUNT_KEYS:
            # Typed at a prompt, so a typo is reported rather than absorbed
            # the way a project file's is.
            try:
                roles[key] = int(value.strip())
            except ValueError:
                raise ValueError(
                    f'--map {key}={value.strip()!r} needs a whole number'
                ) from None
        else:
            valid = ', '.join(sorted(LIST_KEYS | SINGLE_KEYS | COUNT_KEYS))
            raise ValueError(f'unknown mapping key {key!r}; use one of: {valid}')

    return FieldMapping(**roles)
