'''Canon layout, page writes, and wiki-link graph reports.'''

import re

import pytest

from osintgpt.canon import (
    SECTIONS,
    append_log,
    backlinks,
    broken_links,
    create_skeleton,
    links_in,
    page_path,
    page_slug,
    read_page,
    resolve_page,
    write_page
)
from osintgpt.projects import Project


@pytest.fixture
def project(tmp_path):
    return Project.create('Canon Case', home=tmp_path)


def test_a_new_project_has_an_empty_canon_until_asked(project):
    assert project.paths.canon.is_dir()
    assert list(project.paths.canon.iterdir()) == []


def test_the_skeleton_appears_inside_the_existing_directory(project):
    existing = project.paths.canon

    returned = create_skeleton(existing)

    assert returned == existing
    assert (existing / 'index.md').is_file()
    assert (existing / 'log.md').is_file()
    assert all((existing / section).is_dir() for section in SECTIONS)
    assert 'maintained by osintgpt' in (
        existing / 'index.md'
    ).read_text(encoding='utf-8').lower()


def test_creating_the_skeleton_twice_preserves_existing_content(project):
    create_skeleton(project.paths.canon)
    index = project.paths.canon / 'index.md'
    index.write_text('Analyst-edited index.', encoding='utf-8')
    page = write_page(
        project.paths.canon, 'entities', 'Existing', 'Existing page.'
    )

    create_skeleton(project.paths.canon)

    assert index.read_text(encoding='utf-8') == 'Analyst-edited index.'
    assert page.read_text(encoding='utf-8') == 'Existing page.\n'


def test_links_are_extracted_in_order_and_keep_non_latin_targets():
    text = '[[Alpha Corp]] then [[Альфа]] and [[شركة]] and [[分析]].'

    assert links_in(text) == ['Alpha Corp', 'Альфа', 'شركة', '分析']


def test_a_page_name_resolves_to_its_filename(project):
    page = write_page(
        project.paths.canon, 'entities', 'Alpha Corp', 'A page.'
    )

    assert page.name == 'alpha-corp.md'
    assert resolve_page(project.paths.canon, 'Alpha Corp') == page
    assert resolve_page(project.paths.canon, 'entities/Alpha Corp') == page


def test_non_latin_names_make_distinct_nonempty_pages(project):
    names = ('Альфа', 'شركة', '分析')
    pages = {
        write_page(project.paths.canon, 'entities', name, name)
        for name in names
    }

    assert len(pages) == len(names)
    assert all(path.stem for path in pages)
    assert all(
        resolve_page(project.paths.canon, name) in pages for name in names
    )


def test_even_punctuation_only_names_have_a_total_mapping():
    first = page_slug('?!')
    second = page_slug('...')

    assert first.startswith('page-')
    assert second.startswith('page-')
    assert first != second


def test_windows_device_names_do_not_become_reserved_paths():
    assert page_slug('CON').startswith('page-')
    assert page_slug('LPT1').startswith('page-')


def test_a_missing_link_is_reported_without_creating_a_page(project):
    source = write_page(
        project.paths.canon, 'narratives', 'Open question',
        'This points to [[Missing page]].'
    )

    report = broken_links(project.paths.canon)

    assert report == {'Missing page': ['narratives/open-question.md']}
    assert resolve_page(project.paths.canon, 'Missing page') is None
    assert source.is_file()


def test_backlinks_are_found_in_both_directions(project):
    write_page(
        project.paths.canon, 'entities', 'First', 'See [[Second]].'
    )
    write_page(
        project.paths.canon, 'narratives', 'Second', 'See [[First]].'
    )

    report = backlinks(project.paths.canon)

    assert report['narratives/second.md'] == ['entities/first.md']
    assert report['entities/first.md'] == ['narratives/second.md']


def test_a_page_can_link_to_itself_without_recursion(project):
    write_page(
        project.paths.canon, 'decisions', 'Loop', 'See [[Loop]] twice [[Loop]].'
    )

    assert backlinks(project.paths.canon) == {
        'decisions/loop.md': ['decisions/loop.md']
    }


def test_writing_twice_replaces_instead_of_appending(project):
    write_page(project.paths.canon, 'sources', 'Record', 'First version.')
    write_page(project.paths.canon, 'sources', 'Record', 'Second version.')

    text = read_page(project.paths.canon, 'sources', 'Record')

    assert text == 'Second version.\n'
    assert 'First version' not in text


def test_links_passed_to_a_write_are_wiki_links(project):
    write_page(
        project.paths.canon, 'entities', 'Connected', 'Body.',
        links=['Альфа', '分析']
    )

    assert read_page(project.paths.canon, 'entities', 'Connected') == (
        'Body.\n\n[[Альфа]]\n[[分析]]\n'
    )


def test_a_non_latin_page_reads_back_under_the_same_name(project):
    write_page(project.paths.canon, 'entities', 'شركة', 'محتوى الصفحة')

    assert read_page(project.paths.canon, 'entities', 'شركة') == (
        'محتوى الصفحة\n'
    )


def test_reading_a_missing_page_does_not_create_the_skeleton(project):
    assert read_page(project.paths.canon, 'entities', 'Absent') is None
    assert list(project.paths.canon.iterdir()) == []


def test_unknown_sections_are_rejected(project):
    with pytest.raises(ValueError, match='canon section'):
        page_path(project.paths.canon, 'unknown', 'Page')


def test_append_log_only_adds_dated_lines(project):
    create_skeleton(project.paths.canon)
    log = project.paths.canon / 'log.md'
    initial = log.read_text(encoding='utf-8')

    append_log(project.paths.canon, 'created a page')
    after_first = log.read_text(encoding='utf-8')
    append_log(project.paths.canon, 'updated\nthat page')
    after_second = log.read_text(encoding='utf-8')

    assert after_first.startswith(initial)
    assert after_second.startswith(after_first)
    appended = after_second[len(initial):].splitlines()
    assert len(appended) == 2
    assert all(
        re.match(r'- \[\d{4}-\d{2}-\d{2}T', line) for line in appended
    )
    assert appended[1].endswith('updated that page')
