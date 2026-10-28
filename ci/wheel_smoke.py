'''Build committed sources and smoke a venv-local wheel over inherited dependencies.'''

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', required=True, type=Path,
                        help='A new directory outside the checkout; retained as evidence')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scratch = args.scratch.resolve()
    if scratch.is_relative_to(root) or scratch.exists():
        parser.error('--scratch must be new and outside the checkout')
    scratch.mkdir(parents=True)
    working = scratch / 'working'
    working.mkdir()
    for name in ['temp', 'wheels', 'guard/ci']:
        (scratch / name).mkdir(parents=True)
    # Copy only the guards, never the source package, onto the wheel's path.
    for name in ['sitecustomize.py', 'offline_environment.py']:
        shutil.copy2(root / 'ci' / name, scratch / 'guard' / 'ci' / name)
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith(('OSINTGPT_', 'OPENAI_', 'GOOGLE_', 'GEMINI_', 'ANTHROPIC_',
                            'VOYAGE_', 'QDRANT_', 'POSTGRES_', 'OLLAMA_')) or name == 'SQL_DB_FILE_PATH':
            environment.pop(name)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(scratch / 'guard' / 'ci'), str(scratch / 'guard')]),
        OSINTGPT_TEST_SANDBOX=str(scratch / 'sandbox'),
        TEMP=str(scratch / 'temp'), TMP=str(scratch / 'temp'), TMPDIR=str(scratch / 'temp'),
        PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
        PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INDEX='1', PIP_CONFIG_FILE=os.devnull,
    )
    receipts = []

    def run(label, command, cwd=working):
        started = time.monotonic()
        with (scratch / f'{label}.log').open('w', encoding='utf-8') as output:
            result = subprocess.run(command, cwd=cwd, env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        receipts.append({'label': label, 'command': list(map(str, command)), 'cwd': str(cwd),
                         'exit_code': result.returncode, 'wall_seconds': round(time.monotonic() - started, 3)})
        (scratch / 'commands.json').write_text(json.dumps(receipts, indent=2), encoding='utf-8')
        print(label, 'exit:', result.returncode, flush=True)
        if result.returncode:
            print((scratch / f'{label}.log').read_text(encoding='utf-8')[-4000:])
            raise SystemExit(result.returncode)

    run('revision', ['git', 'rev-parse', 'HEAD'], root)
    archive = scratch / 'source.zip'
    run('source-archive', ['git', 'archive', '--format=zip', f'--output={archive}', 'HEAD'], root)
    source = scratch / 'source'
    with ZipFile(archive) as zipped:
        zipped.extractall(source)
    run('build', [sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-index',
                  '--no-build-isolation', '--no-cache-dir', '--wheel-dir', str(scratch / 'wheels'), str(source)])
    wheels = list((scratch / 'wheels').glob('osintgpt-*.whl'))
    assert len(wheels) == 1, wheels
    wheel = wheels[0]
    # Check every surviving package source and asset and reject retired packages.
    expected = [path.relative_to(source).as_posix() for path in (source / 'osintgpt').rglob('*')
                if path.is_file() and (path.suffix == '.py' or '/assets/' in path.as_posix()
                                       or '/prompts/templates/' in path.as_posix())]
    with ZipFile(wheel) as zipped:
        retired = ('osintgpt/llms/', 'osintgpt/embeddings/',
                   'osintgpt/semantic_operations/', 'osintgpt/databases/')
        assert not any(name.startswith(retired) for name in zipped.namelist())
        for name in expected:
            assert zipped.read(name) == (source / name).read_bytes(), name
    (scratch / 'wheel-contents.json').write_text(json.dumps({
        'wheel': wheel.name, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'verified_files': sorted(expected),
    }, indent=2), encoding='utf-8')
    venv = scratch / 'venv'
    run('create-venv', [sys.executable, '-m', 'venv', '--system-site-packages', str(venv)])
    scripts = venv / ('Scripts' if os.name == 'nt' else 'bin')
    python = scripts / ('python.exe' if os.name == 'nt' else 'python')
    run('install', [str(python), '-m', 'pip', 'install', '--no-deps', '--no-index',
                    '--ignore-installed', str(wheel)])
    run('cli-help', [str(scripts / ('osintgpt.exe' if os.name == 'nt' else 'osintgpt')), '--help'])
    probe = working / 'wheel_probe.py'
    shutil.copy2(source / 'ci' / 'wheel_probe.py', probe)
    run('installed-probe', [str(python), str(probe)])
    print((scratch / 'installed-probe.log').read_text(encoding='utf-8'))
    print('Evidence retained:', scratch)


if __name__ == '__main__':
    main()
