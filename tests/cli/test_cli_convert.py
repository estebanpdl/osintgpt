# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: test_cli_convert.py
# Description: The convert command through Typer's public surface, and the
#   promise it rests on — no project, no model, no network unless asked.
# =================================================================================

import json

import pytest
from typer.testing import CliRunner

from osintgpt.cli import app
from osintgpt.cli import convert as cli_convert
from osintgpt.projects import Registry


@pytest.fixture
def home(tmp_path):
    return tmp_path / 'home'


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def note(tmp_path):
    path = tmp_path / 'note.md'
    path.write_text('# Title\n\nUn párrafo del expediente.\n', encoding='utf-8')

    return path


@pytest.fixture
def records(tmp_path):
    path = tmp_path / 'registros.csv'
    path.write_text(
        'record_id,body\nr1,primer registro\nr2,segundo registro\n',
        encoding='utf-8'
    )

    return path


@pytest.fixture
def folder(tmp_path, note, records):
    root = tmp_path / 'material'
    (root / 'sub').mkdir(parents=True)
    (root / 'report.txt').write_text('Prose at the top.', encoding='utf-8')
    (root / 'sub' / 'nested.md').write_text('# Nested', encoding='utf-8')
    (root / 'capture.bin').write_bytes(b'\x00\x01')

    return root


def invoke(runner, home, *arguments):
    return runner.invoke(app, ['--home', str(home), *arguments])


class TestOutput:
    def test_markdown_goes_to_standard_output(self, runner, home, note):
        result = invoke(runner, home, 'convert', str(note))

        assert result.exit_code == 0
        assert '# Title' in result.stdout
        assert 'Un párrafo del expediente.' in result.stdout

    def test_what_it_took_stays_off_standard_output(
        self, runner, home, note
    ):
        '''
        The result has to be pipeable. A note about the reader on stdout would
        ride into whatever the operator redirects this into.
        '''
        result = runner.invoke(
            app, ['--home', str(home), 'convert', str(note)]
        )

        assert 'read by text' not in result.stdout
        assert 'read by text' in result.stderr

    def test_out_writes_the_file_and_prints_no_markdown(
        self, runner, home, note, tmp_path
    ):
        target = tmp_path / 'out' / 'note.md'

        result = invoke(runner, home, 'convert', str(note), '--out', str(target))

        assert result.exit_code == 0
        assert target.read_text(encoding='utf-8').startswith('# Title')
        assert '# Title' not in result.stdout


class TestDestinations:
    '''
    One file and a destination directory is an ordinary thing to type, and
    writing the markdown to the directory itself reports as a denied
    permission rather than as the mistake it is.
    '''

    def test_an_existing_directory_receives_the_file(
        self, runner, home, note, tmp_path
    ):
        out = tmp_path / 'converted'
        out.mkdir()

        result = invoke(runner, home, 'convert', str(note), '--out', str(out))

        assert result.exit_code == 0
        assert (out / 'note.md.md').read_text(encoding='utf-8').startswith(
            '# Title'
        )

    def test_a_trailing_separator_names_a_directory_that_is_not_there_yet(
        self, runner, home, note, tmp_path
    ):
        out = tmp_path / 'converted'

        result = invoke(
            runner, home, 'convert', str(note), '--out', f'{out}/'
        )

        assert result.exit_code == 0
        assert (out / 'note.md.md').is_file()

    def test_any_other_path_is_the_file_to_write(
        self, runner, home, note, tmp_path
    ):
        target = tmp_path / 'converted' / 'chosen-name.md'

        result = invoke(
            runner, home, 'convert', str(note), '--out', str(target)
        )

        assert result.exit_code == 0
        assert target.is_file()
        assert not (target.parent / 'note.md.md').exists()

    def test_the_written_path_is_named_back(
        self, runner, home, note, tmp_path
    ):
        '''
        A destination directory decides the filename; the operator has not
        seen it yet.
        '''
        out = tmp_path / 'converted'
        out.mkdir()

        result = invoke(runner, home, 'convert', str(note), '--out', str(out))

        assert 'note.md.md' in result.stdout

    def test_a_write_that_cannot_succeed_is_a_message_not_a_traceback(
        self, runner, home, note, tmp_path
    ):
        blocker = tmp_path / 'blocker'
        blocker.write_text('not a directory', encoding='utf-8')

        result = invoke(
            runner, home, 'convert', str(note),
            '--out', str(blocker / 'inner.md')
        )

        assert result.exit_code == 1
        assert not isinstance(result.exception, OSError)
        assert 'could not write' in result.output

    def test_json_carries_the_markdown_and_the_reader(
        self, runner, home, note
    ):
        result = invoke(runner, home, 'convert', str(note), '--json')
        data = json.loads(result.stdout)

        assert data['converted'][0]['reader'] == 'text'
        assert data['converted'][0]['documents'] == 1
        assert '# Title' in data['converted'][0]['markdown']


class TestFolders:
    def test_a_directory_needs_somewhere_to_write(self, runner, home, folder):
        result = invoke(runner, home, 'convert', str(folder))

        assert result.exit_code != 0
        assert '--out <directory>' in result.output

    def test_a_destination_that_is_a_file_is_refused(
        self, runner, home, folder, note
    ):
        result = invoke(
            runner, home, 'convert', str(folder), '--out', str(note)
        )

        assert result.exit_code != 0
        assert 'must be a directory' in result.output

    def test_a_destination_written_like_a_filename_is_refused(
        self, runner, home, folder, tmp_path
    ):
        '''
        This used to create a directory called all.md and fill it, which
        reads as a bug rather than as a decision.
        '''
        out = tmp_path / 'converted' / 'all.md'

        result = invoke(
            runner, home, 'convert', str(folder), '--out', str(out)
        )

        assert result.exit_code != 0
        assert 'names a file' in result.output
        assert not out.exists()

    def test_a_trailing_separator_takes_the_name_as_a_directory(
        self, runner, home, folder, tmp_path
    ):
        out = tmp_path / 'v1.2'

        result = invoke(
            runner, home, 'convert', str(folder), '--out', f'{out}/'
        )

        assert result.exit_code == 0
        assert (out / 'report.txt.md').is_file()

    def test_a_destination_that_is_not_there_yet_is_created(
        self, runner, home, folder, tmp_path
    ):
        out = tmp_path / 'converted'

        result = invoke(
            runner, home, 'convert', str(folder), '--out', str(out)
        )

        assert result.exit_code == 0
        assert (out / 'report.txt.md').is_file()

    def test_each_file_keeps_its_place_and_its_whole_name(
        self, runner, home, folder, tmp_path
    ):
        out = tmp_path / 'converted'

        result = invoke(
            runner, home, 'convert', str(folder), '--out', str(out)
        )

        assert result.exit_code == 0
        assert (out / 'report.txt.md').read_text(encoding='utf-8') == (
            'Prose at the top.'
        )
        assert (out / 'sub' / 'nested.md.md').is_file()

    def test_two_files_of_one_name_do_not_converge(
        self, runner, home, tmp_path
    ):
        root = tmp_path / 'both'
        root.mkdir()
        (root / 'report.txt').write_text('From the text file.', encoding='utf-8')
        (root / 'report.md').write_text('From the markdown.', encoding='utf-8')
        out = tmp_path / 'converted'

        invoke(runner, home, 'convert', str(root), '--out', str(out))

        assert (out / 'report.txt.md').is_file()
        assert (out / 'report.md.md').is_file()

    def test_what_nothing_reads_is_reported_rather_than_dropped(
        self, runner, home, folder, tmp_path
    ):
        result = invoke(
            runner, home, 'convert', str(folder),
            '--out', str(tmp_path / 'converted'), '--json'
        )
        data = json.loads(result.stdout)

        assert [
            path for path in data['unsupported'] if path.endswith('capture.bin')
        ]

    def test_one_unreadable_file_does_not_stop_the_rest(
        self, runner, home, folder, records, tmp_path
    ):
        '''
        A folder holds the file that needs a decision. Stopping on it would
        mean converting the other two hundred again afterwards.
        '''
        (folder / 'registros.csv').write_text(
            records.read_text(encoding='utf-8'), encoding='utf-8'
        )
        out = tmp_path / 'converted'

        result = invoke(
            runner, home, 'convert', str(folder), '--out', str(out), '--json'
        )
        data = json.loads(result.stdout)

        assert result.exit_code == 1
        assert len(data['converted']) == 2
        assert data['failed'][0]['path'].endswith('registros.csv')


class TestStructured:
    def test_a_structured_file_is_not_guessed_at(self, runner, home, records):
        result = invoke(runner, home, 'convert', str(records))

        assert result.exit_code != 0
        assert 'record_id' in result.output
        assert 'body' in result.output
        assert '--map content=<field>' in result.output

    def test_a_mapping_makes_one_section_per_record(
        self, runner, home, records
    ):
        result = invoke(
            runner, home, 'convert', str(records),
            '--map', 'content=body', '--map', 'identity=record_id'
        )

        assert result.exit_code == 0
        assert result.stdout.count('## ') == 2
        assert 'primer registro' in result.stdout

    def test_a_bad_mapping_key_is_reported(self, runner, home, records):
        result = invoke(
            runner, home, 'convert', str(records), '--map', 'cuerpo=body'
        )

        assert result.exit_code != 0
        assert 'unknown mapping key' in result.output


class TestRefusals:
    def test_an_image_says_why_it_cannot_be_converted(
        self, runner, home, tmp_path
    ):
        path = tmp_path / 'scan.png'
        path.write_bytes(b'\x89PNG\r\n')

        result = invoke(runner, home, 'convert', str(path))

        assert result.exit_code != 0
        assert 'images are embedded' in result.output

    def test_a_missing_path_is_named(self, runner, home, tmp_path):
        result = invoke(runner, home, 'convert', str(tmp_path / 'absent.md'))

        assert result.exit_code != 0
        assert 'no such path' in result.output


class TestVision:
    def test_converting_builds_no_provider(
        self, runner, home, note, monkeypatch
    ):
        '''
        The whole point of the default: a file becomes markdown without a
        project, a credential, or a call to anything.
        '''
        def refuse(*arguments, **keywords):
            raise AssertionError('a provider was built without --vision')

        monkeypatch.setattr(cli_convert, 'build_generation_provider', refuse)
        monkeypatch.setattr(cli_convert, 'resolve_credentials', refuse)

        result = invoke(runner, home, 'convert', str(note))

        assert result.exit_code == 0

    def test_vision_without_a_model_anywhere_says_so(
        self, runner, home, note
    ):
        result = invoke(runner, home, 'convert', str(note), '--vision')

        assert result.exit_code != 0
        assert '--model' in result.output

    def test_a_named_model_is_the_one_built(
        self, runner, home, note, monkeypatch
    ):
        built = {}

        def record(provider, settings, model=None, recorder=None):
            built['provider'] = provider
            built['model'] = model

            return object()

        monkeypatch.setattr(cli_convert, 'build_generation_provider', record)

        result = invoke(
            runner, home, 'convert', str(note), '--model', 'gpt-4.1-mini'
        )

        assert result.exit_code == 0
        assert built['model'] == 'gpt-4.1-mini'

    def test_a_scanned_page_is_read_and_cached_under_the_project(
        self, runner, home, tmp_path, monkeypatch
    ):
        '''
        The cache is the reason --project is worth passing: a page read while
        exploring must not be paid for again at index time.
        '''
        pypdfium2 = pytest.importorskip('pypdfium2')
        path = tmp_path / 'scanned.pdf'
        document = pypdfium2.PdfDocument.new()
        document.new_page(595, 842).gen_content()
        document.save(str(path))
        document.close()

        class Stub:
            def describe_image(self, system, user, image,
                               media_type='image/png'):
                return 'Transcribed from the scan.'

        monkeypatch.setattr(
            cli_convert, 'build_generation_provider',
            lambda *arguments, **keywords: Stub()
        )
        assert invoke(runner, home, 'project', 'create', 'Caso A').exit_code == 0

        result = invoke(
            runner, home, 'convert', str(path), '--project', 'caso-a',
            '--model', 'gpt-4.1-mini'
        )
        project = Registry.load(home).open('caso-a')

        assert result.exit_code == 0
        assert 'Transcribed from the scan.' in result.stdout
        assert list(project.paths.extracts.glob('*.md'))
