# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: credentials.py
# Description: Where an operator's API keys are kept when they are not in the
#   environment, and the order the two are consulted in. Nothing is discovered:
#   the home is always an argument.
# =================================================================================

# import modules
import os
import tomli_w

# import submodules
from dataclasses import dataclass
from pathlib import Path

# type hints
from typing import Dict, List, Optional, Union

# import osintgpt config
from osintgpt.config import ENV_VARS, Settings, secret_fields

# import osintgpt projects
from osintgpt.projects.toml_io import read_toml

CREDENTIALS_FILE = 'credentials.toml'
SECTION = 'credentials'

CREDENTIALS_HEADER = '''\
# osintgpt credentials
#
# Written by `osintgpt auth set`. Values are stored in plain text: this file is
# a convenience over exporting variables, not a secret manager. Keep it off
# shared and synced drives, and prefer the environment on a machine you do not
# control.

'''

ENVIRONMENT = 'environment'
STORED = 'stored'


# CredentialStatus class
@dataclass(frozen=True)
class CredentialStatus:
    '''
    What is known about one credential without revealing it: which provider it
    belongs to, whether it is set, and which source would win.
    '''
    provider: str
    field: str
    variable: str
    source: Optional[str] = None
    # True when a variable in the environment is overriding a stored value.
    # Worth surfacing on its own: the operator set one key and a different one
    # is being used, which no amount of re-running `auth set` would explain.
    shadowed: bool = False

    @property
    def is_set(self) -> bool:
        return self.source is not None


# provider names derived from the settings fields
def credential_names() -> Dict[str, str]:
    '''
    Map each provider name an operator would type to the setting it fills.

    Derived from the field names rather than listed, so a credential added to
    Settings later is offered here without anyone remembering to add it.

    Returns:
        Dict[str, str]: Provider name to settings field.
    '''
    names = {}
    for field in secret_fields():
        for suffix in ('_api_key', '_dsn'):
            if field.endswith(suffix):
                names[field[:-len(suffix)]] = field
                break

    return names


# resolve what an operator typed into a settings field
def field_for(name: str) -> Optional[str]:
    '''
    Accept either the provider name or the settings field, because both are
    things an operator reasonably types and neither is a guess.

    Args:
        name (str): Provider name, settings field, or environment variable.

    Returns:
        Optional[str]: The settings field, or None when unrecognised.
    '''
    cleaned = (name or '').strip().lower()
    names = credential_names()
    if cleaned in names:
        return names[cleaned]
    if cleaned in secret_fields():
        return cleaned

    for field, variable in ENV_VARS.items():
        if cleaned == variable.lower() and field in secret_fields():
            return field

    return None


# where credentials live for a home
def credentials_file(home: Union[str, Path]) -> Path:
    '''
    Args:
        home (Union[str, Path]): The osintgpt home.

    Returns:
        Path: The credentials file for that home.
    '''
    return Path(home) / CREDENTIALS_FILE


# read stored credentials
def load_credentials(home: Union[str, Path]) -> Dict[str, str]:
    '''
    Read the credentials stored for a home.

    Args:
        home (Union[str, Path]): The osintgpt home.

    Returns:
        Dict[str, str]: Settings field to value, empty when nothing is \
            stored. Unknown keys are dropped rather than carried, so a file \
            written by a newer osintgpt opens in an older one.
    '''
    document = read_toml(credentials_file(home))
    stored = document.get(SECTION, {}) or {}
    known = secret_fields()

    return {
        field: value for field, value in stored.items()
        if field in known and isinstance(value, str) and value
    }


# replace stored credentials
def save_credentials(
    home: Union[str, Path], values: Dict[str, str]
) -> Path:
    '''
    Write the credentials file, replacing whatever was there.

    Args:
        home (Union[str, Path]): The osintgpt home.
        values (Dict[str, str]): Settings field to value. An empty mapping \
            removes the file rather than leaving an empty one, so "no \
            credentials stored" has one representation.

    Returns:
        Path: The file written, or would have been.
    '''
    path = credentials_file(home)
    if not values:
        path.unlink(missing_ok=True)

        return path

    document = CREDENTIALS_HEADER + tomli_w.dumps({SECTION: dict(values)})
    _write_private(path, document)

    return path


# store one credential
def store_credential(
    home: Union[str, Path], field: str, value: str
) -> Path:
    '''
    Args:
        home (Union[str, Path]): The osintgpt home.
        field (str): Settings field to write.
        value (str): The credential.

    Raises:
        ValueError: If the field is not a credential, or the value is empty.

    Returns:
        Path: The file written.
    '''
    if field not in secret_fields():
        raise ValueError(f'{field} is not a credential')

    cleaned = (value or '').strip()
    if not cleaned:
        raise ValueError('a credential cannot be empty')

    values = load_credentials(home)
    values[field] = cleaned

    return save_credentials(home, values)


# forget one credential
def remove_credential(home: Union[str, Path], field: str) -> bool:
    '''
    Args:
        home (Union[str, Path]): The osintgpt home.
        field (str): Settings field to forget.

    Returns:
        bool: True when something was removed.
    '''
    values = load_credentials(home)
    if field not in values:
        return False

    del values[field]
    save_credentials(home, values)

    return True


# settings built from the environment and the stored credentials
def resolve_credentials(
    home: Union[str, Path], **overrides
) -> Settings:
    '''
    Build Settings from the process environment, filling gaps from the
    credentials file.

    The environment wins. That is the convention every other tool with a
    credentials file follows, and it is what keeps a container, a CI job or a
    one-off shell able to override a stored key without editing a file. The
    cost is that a stale variable can silently shadow a freshly stored one,
    which is why `credential_status` reports shadowing rather than leaving the
    operator to discover it.

    Args:
        home (Union[str, Path]): The osintgpt home.
        **overrides: Field values that win over both sources.

    Returns:
        Settings: Ready to hand to a provider.
    '''
    environment = Settings.from_env()
    stored = {
        field: value for field, value in load_credentials(home).items()
        if not getattr(environment, field, '')
    }
    stored.update(overrides)

    return environment.with_overrides(**stored) if stored else environment


# what is set, and where it came from
def credential_status(home: Union[str, Path]) -> List[CredentialStatus]:
    '''
    Report every credential's presence and source, never its value.

    Args:
        home (Union[str, Path]): The osintgpt home.

    Returns:
        List[CredentialStatus]: One row per provider, by name.
    '''
    environment = Settings.from_env()
    stored = load_credentials(home)

    rows = []
    for provider, field in sorted(credential_names().items()):
        in_environment = bool(getattr(environment, field, ''))
        in_file = field in stored
        source = ENVIRONMENT if in_environment else STORED if in_file else None
        rows.append(CredentialStatus(
            provider=provider,
            field=field,
            variable=ENV_VARS.get(field, field.upper()),
            source=source,
            shadowed=in_environment and in_file
        ))

    return rows


# write a file only its owner can read, where the platform allows it
def _write_private(path: Path, document: str) -> None:
    '''
    Create the file with restrictive permissions and write it.

    The mode is set at creation rather than after: writing first and narrowing
    afterwards leaves a window where the credential is readable by anyone, and
    that window is exactly long enough to matter on a shared machine.

    POSIX honours the mode. Windows does not model permissions this way, so
    there the file stays readable by the account that wrote it — which the
    file's own header says plainly rather than implying a protection that is
    not there.
    '''
    path.parent.mkdir(parents=True, exist_ok=True)
    # An existing file keeps its own mode through O_TRUNC, so replace it.
    path.unlink(missing_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(document)
