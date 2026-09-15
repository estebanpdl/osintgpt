'''Supported files as markdown. Free by default; vision is asked for.'''

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer

from osintgpt.credentials import resolve_credentials
from osintgpt.exceptions.errors import MissingEnvironmentVariableError
from osintgpt.ingestion import Conversion, to_markdown
from osintgpt.ingestion.loaders import READABLE_SUFFIXES, needs_mapping
from osintgpt.ingestion.preview import walk_files
from osintgpt.ingestion.tabular import UnmappedSourceError, describe_fields
from osintgpt.ingestion.transcription import transcriber_for
from osintgpt.llm import build_generation_provider
from osintgpt.llm.usage import CostLimitReached
from osintgpt.projects import load_user_defaults
from osintgpt.projects.paths import EXTRACTS_DIR

from .costs import fail_for_cost, recorder_for, render_usage, usage_data
from .mapping import MAPPING_EXAMPLE, build_mapping
from .output import emit, error_console, fail
from .selection import ProjectSelectionError, resolve_project, state_from

UNMAPPED = 'needs a content field mapping'


def convert_file(
    context: typer.Context,
    path: Path = typer.Argument(..., help='File or directory to convert.'),
    out: Optional[str] = typer.Option(
        None, '--out', metavar='PATH',
        help='When converting a single file, --out specifies the output '
             'filename; when converting a directory, --out must be an '
             'existing directory where the resulting .md files will be '
             'written, preserving the original structure.'
    ),
    maps: Optional[List[str]] = typer.Option(
        None, '--map',
        help='Field role as key=value; repeatable. Required for structured '
             'files, same keys as `osintgpt add`.'
    ),
    vision: bool = typer.Option(
        False, '--vision',
        help='Transcribe PDF pages that hold no extractable text. Costs one '
             'generation call per page. Off by default'
    ),
    model: Optional[str] = typer.Option(
        None, '--model',
        help='Model that transcribes scanned pages. Implies --vision.'
    ),
    project_slug: Optional[str] = typer.Option(
        None, '--project',
        help='Take the transcription model and cache from this project. Not '
             'needed to convert.'
    ),
    json_output: bool = typer.Option(False, '--json', help='Print JSON only.')
) -> None:
    if not path.exists():
        fail(f'no such path: {path}', json_output)

    if path.is_dir():
        if out is None:
            fail(
                f'{path} is a directory; pass --out <directory> to write the '
                'markdown somewhere', json_output
            )
        refusal = _unusable_destination(out)
        if refusal:
            fail(refusal, json_output)

    try:
        mapping = build_mapping(maps)
    except ValueError as error:
        fail(str(error), json_output)

    transcriber = recorder = None
    if vision or model:
        transcriber, recorder = _vision(
            state_from(context), project_slug, model, json_output
        )

    files: List[Path] = []
    skipped: List[Path] = []
    for found in walk_files(path):
        readable = found.suffix.lower() in READABLE_SUFFIXES
        (files if readable else skipped).append(found)

    if not files:
        fail(f'nothing readable under {path}', json_output)

    results: List[Conversion] = []
    problems: List[Tuple[Path, str]] = []
    try:
        for found in files:
            conversion, problem = _convert_one(found, mapping, transcriber)
            if conversion is None:
                if path.is_file():
                    _fail_for(found, problem, json_output)
                problems.append((found, problem))
                continue
            results.append(conversion)
    except CostLimitReached as error:
        fail_for_cost(error, recorder, json_output)

    written = _write(results, path, out, json_output)
    _report(
        results, problems, skipped, written, out, recorder, json_output,
        single=path.is_file()
    )

    if problems:
        raise typer.Exit(code=1)


def _convert_one(
    path: Path, mapping, transcriber
) -> Tuple[Optional[Conversion], str]:
    '''
    One file's markdown, or why there is none.

    A directory of documents holds the file nothing can read, and stopping the
    walk on it would mean converting the other two hundred again afterwards.
    '''
    if needs_mapping(path) and not mapping.is_set:
        return None, UNMAPPED

    try:
        return to_markdown(path, mapping, transcriber), ''
    except CostLimitReached:
        raise
    except UnmappedSourceError:
        return None, UNMAPPED
    except Exception as error:  # noqa: BLE001 — one file, not the walk
        return None, str(error)


def _fail_for(path: Path, problem: str, json_output: bool) -> None:
    if problem != UNMAPPED:
        fail(f'{path.name}: {problem}', json_output)

    fail(
        f'{path.name} {UNMAPPED}', json_output,
        {
            'fields': describe_fields(path),
            'try': f'osintgpt convert "{path}" {MAPPING_EXAMPLE}'
        }
    )


def _vision(state, project_slug: Optional[str], model: Optional[str],
            json_output: bool):
    '''
    A transcriber and the recorder watching what it spends.

    A project is optional here and supplies two things when given: which model
    reads a page, and a cache under its own `extracts/` so a page read while
    exploring is not paid for again at index time.
    '''
    home = Path(state.home)
    defaults = load_user_defaults(home)
    project = None

    if project_slug:
        try:
            project = resolve_project(home, project_slug)
        except ProjectSelectionError as error:
            fail(str(error), json_output)

    settings = project.effective_settings(defaults) if project else defaults
    credentials = resolve_credentials(home)
    if project:
        credentials = project.settings_for(credentials, defaults)

    name = model or settings.ingestion_model or settings.generation_model
    if not name:
        fail(
            'no model to transcribe with: pass --model <name>, or set one '
            'with `osintgpt config set ingestion_model <name>`',
            json_output
        )

    recorder = recorder_for(settings)
    try:
        generator = build_generation_provider(
            settings.generation_provider, credentials, model=name,
            recorder=recorder
        )
    except (ImportError, MissingEnvironmentVariableError, ValueError) as error:
        fail(str(error), json_output, {'usage': usage_data(recorder)})

    cache = project.paths.extracts if project else home / EXTRACTS_DIR

    return transcriber_for(generator, cache), recorder


def _write(
    results: List[Conversion],
    root: Path,
    out: Optional[str],
    json_output: bool
) -> List[Path]:
    if out is None:
        return []

    if root.is_file():
        conversion = results[0]
        target = _target(out, conversion.path)

        return [_save(conversion.markdown, target, json_output)]

    destination = Path(out)
    written: List[Path] = []
    for conversion in results:
        # The original name is kept whole, suffix included: a directory holding
        # both report.pdf and report.docx would otherwise converge on one file
        # and lose whichever was written first.
        target = destination / conversion.path.relative_to(root).parent / (
            f'{conversion.path.name}.md'
        )
        written.append(_save(conversion.markdown, target, json_output))

    return written


def _unusable_destination(out: str) -> str:
    '''
    Why a converted directory cannot be written here, if it cannot.

    An absent destination is created, unless it is written like a filename:
    `--out markdown/all.md` used to make a directory called all.md and fill
    it, which reads as a bug rather than as a decision.
    '''
    destination = Path(out)
    if destination.is_dir() or out.endswith(('/', '\\')):
        return ''

    if not destination.exists() and not destination.suffix:
        return ''

    return (
        'converting a directory writes one .md file per document, so --out '
        f'must be a directory. {destination} names a file; pass a directory, '
        'or end the path with a separator to have it created'
    )


def _target(out: str, source: Path) -> Path:
    '''
    Where one file's markdown goes.

    A destination that is already a directory, or written with a trailing
    separator, names the place rather than the file: writing to it directly
    opens the directory, which Windows reports as a denied permission.
    '''
    destination = Path(out)
    if destination.is_dir() or out.endswith(('/', '\\')):
        return destination / f'{source.name}.md'

    return destination


def _save(markdown: str, target: Path, json_output: bool) -> Path:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown, encoding='utf-8')
    except OSError as error:
        fail(f'could not write {target}: {error}', json_output)

    return target


def _report(
    results: List[Conversion],
    problems: List[Tuple[Path, str]],
    skipped: List[Path],
    written: List[Path],
    out: Optional[Path],
    recorder,
    json_output: bool,
    single: bool
) -> None:
    data: Dict[str, Any] = {
        'converted': [
            {
                'path': conversion.path.as_posix(),
                'reader': conversion.reader,
                'documents': conversion.documents,
                'pages': conversion.pages,
                'transcribed_pages': conversion.transcribed_pages,
                'empty_pages': conversion.empty_pages,
                'markdown': conversion.markdown
            }
            for conversion in results
        ],
        'written': [target.as_posix() for target in written],
        'failed': [
            {'path': path.as_posix(), 'problem': problem}
            for path, problem in problems
        ],
        'unsupported': [path.as_posix() for path in skipped]
    }
    if recorder is not None:
        data['usage'] = usage_data(recorder)

    def render(target) -> None:
        if out is None:
            # Markdown alone on standard output, so the result can be piped or
            # redirected. What it took to produce goes to standard error.
            _echo(results[0].markdown)
        elif len(written) == 1:
            # Named rather than counted: a destination directory decides the
            # filename, and the operator has not seen it yet.
            target.print(f'written to {written[0]}')
        else:
            target.print(f'{len(written)} file(s) written to {out}')

        _notes(results, problems, skipped, single, recorder is not None)
        if recorder is not None:
            render_usage(target, recorder)

    emit(data, json_output, render)


def _notes(
    results: List[Conversion],
    problems: List[Tuple[Path, str]],
    skipped: List[Path],
    single: bool,
    transcribing: bool
) -> None:
    if single and results:
        conversion = results[0]
        detail = f'{conversion.documents} document(s)'
        if conversion.pages:
            detail += f', {conversion.pages} page(s)'
        error_console.print(
            f'{conversion.path.name}: read by {conversion.reader}, {detail}',
            style='dim'
        )

    unread = sum(conversion.empty_pages for conversion in results)
    if unread:
        recover = (
            'the transcription came back empty or failed' if transcribing
            else '--model <name> transcribes them at one generation call each'
        )
        error_console.print(
            f'{unread} page(s) hold no extractable text; {recover}',
            style='yellow'
        )

    if skipped:
        error_console.print(
            f'{len(skipped)} file(s) skipped: osintgpt reads none of those '
            'extensions',
            style='dim'
        )

    for path, problem in problems:
        error_console.print(f'{path.name}: {problem}', style='bold red')


def _echo(text: str) -> None:
    '''
    Written as bytes: a console encoding narrower than the document would
    otherwise fail on the way out, having already read the file successfully.
    '''
    buffer = getattr(sys.stdout, 'buffer', None)
    if buffer is None:
        typer.echo(text)

        return

    buffer.write(text.encode('utf-8') + b'\n')
    buffer.flush()


def register_convert_command(app: typer.Typer) -> None:
    app.command(
        'convert',
        help='Convert a supported file to markdown. No project, no model, no '
             'network unless --vision is asked for.'
    )(convert_file)
