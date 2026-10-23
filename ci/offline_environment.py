'''Isolate test configuration before application imports and in child Python runs.'''

import json
import os
from pathlib import Path
import sys
import tempfile


SANDBOX_VARIABLE = 'OSINTGPT_TEST_SANDBOX'
PROTECTED_VARIABLE = 'OSINTGPT_TEST_PROTECTED_PATHS'
_CONTROL_VARIABLES = {SANDBOX_VARIABLE, PROTECTED_VARIABLE}
_PROVIDER_PREFIXES = (
    'OPENAI_', 'GEMINI_', 'GOOGLE_', 'ANTHROPIC_', 'VOYAGE_', 'QDRANT_',
    'POSTGRES_', 'OLLAMA_', 'OSINTGPT_',
)


def clear_configuration(environ):
    '''Discard inherited provider settings, including service-contract opt-ins.'''
    for name in list(environ):
        if name not in _CONTROL_VARIABLES and (
            name.startswith(_PROVIDER_PREFIXES) or name == 'SQL_DB_FILE_PATH'
        ):
            environ.pop(name, None)


def home_environment(home):
    '''Variables used by pathlib, SDKs and platform configuration directories.'''
    home = Path(home)
    return {
        'HOME': str(home), 'USERPROFILE': str(home),
        'APPDATA': str(home / 'appdata'),
        'LOCALAPPDATA': str(home / 'localappdata'),
        'XDG_CONFIG_HOME': str(home / 'config'),
        'XDG_CACHE_HOME': str(home / 'cache'),
    }


def _normalized(path):
    return os.path.normcase(os.path.abspath(os.fsdecode(path)))


class UserStateBlocked(RuntimeError):
    pass


def protect_user_state(paths):
    '''Refuse Python file/SQLite opens under original application homes.

    Lexical checks run before realpath so resolving an obvious protected path
    does not traverse it. This is a Python I/O guard, not an OS sandbox.
    '''
    protected = tuple(_normalized(path) for path in paths)
    installed = getattr(sys, '_osintgpt_protected_paths', set())
    if set(protected) <= installed:
        return
    sys._osintgpt_protected_paths = installed | set(protected)

    def is_protected(path):
        return any(path == root or path.startswith(root + os.sep) for root in protected)

    def audit(event, args):
        if event not in ('open', 'sqlite3.connect') or not args:
            return
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        path = os.fsdecode(path)
        if event == 'sqlite3.connect' and path.startswith('file:'):
            from urllib.parse import unquote, urlsplit
            path = unquote(urlsplit(path).path)
            if os.name == 'nt' and path.startswith('/') and len(path) > 2 and path[2] == ':':
                path = path[1:]
        lexical = _normalized(path)
        if os.path.basename(lexical) in ('.env', 'credentials.toml'):
            sandbox = _normalized(os.environ[SANDBOX_VARIABLE])
            if not lexical.startswith(sandbox + os.sep):
                raise UserStateBlocked('test attempted to open configuration outside its sandbox')
        if is_protected(lexical) or is_protected(_normalized(os.path.realpath(lexical))):
            raise UserStateBlocked('test attempted to open real osintgpt user state')

    sys.addaudithook(audit)


def isolate_environment():
    '''Return a disposable session root; child processes inherit the guards.'''
    original_homes = json.loads(os.environ.get(PROTECTED_VARIABLE, '[]'))
    if not original_homes:
        original_homes = [str(Path.home() / '.osintgpt')]
    root = os.environ.get(SANDBOX_VARIABLE)
    if root is None:
        root = tempfile.mkdtemp(prefix='osintgpt-tests-')
    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    os.environ[SANDBOX_VARIABLE] = str(root)
    os.environ[PROTECTED_VARIABLE] = json.dumps(original_homes)
    clear_configuration(os.environ)
    os.environ.update(home_environment(root / 'home'))
    protect_user_state(original_homes)

    # Keep the worktree and startup guard ahead of any installed package in
    # child interpreters, including xdist workers launched outside the root.
    repo = Path(__file__).resolve().parents[1]
    inherited = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    paths = [str(repo / 'ci'), str(repo), *filter(None, inherited)]
    os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    return root
