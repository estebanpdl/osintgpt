'''Capture pytest output and exit status, including a failing run's final summary.'''

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.pytest_args
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    command = [sys.executable, '-m', 'pytest', *arguments]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # File handles avoid PowerShell native-stderr pipelines turning pytest's
    # nonzero exit into an exception before the summary has been saved.
    with args.log.open('w', encoding='utf-8') as output:
        environment = dict(os.environ, PYTHONIOENCODING='utf-8')
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=environment)
    receipt = {
        'command': command, 'cwd': str(Path.cwd()),
        'exit_code': result.returncode,
        'wall_seconds': round(time.monotonic() - started, 3),
        'log': str(args.log.absolute()),
    }
    args.log.with_suffix('.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(args.log.read_text(encoding='utf-8'), end='')
    print(json.dumps(receipt), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
