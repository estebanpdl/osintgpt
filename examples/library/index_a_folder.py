"""Create or load a project, register one folder, and index it.

Usage:
    python examples/library/index_a_folder.py PROJECT FOLDER [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, index_project
from osintgpt.ingestion import Corpus
from osintgpt.llm import build_embedding_provider


def load_or_create(path: Path, name: str | None) -> Project:
    """Open ``path`` as a project, creating it when needed."""
    if (path / 'project.toml').is_file():
        return Project.load(path)

    return Project.create(name or path.name, path=path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Register and index one folder in a project.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('folder', type=Path, help='folder to register')
    parser.add_argument('--name', help='display name when creating the project')
    parser.add_argument('--embedding-provider', help='provider id')
    parser.add_argument('--embedding-model', help='model name')
    parser.add_argument('--force', action='store_true', help='re-index all files')
    arguments = parser.parse_args()

    project = load_or_create(arguments.project.resolve(), arguments.name)
    changes = {}
    if arguments.embedding_provider:
        changes['embedding_provider'] = arguments.embedding_provider
    if arguments.embedding_model:
        changes['embedding_model'] = arguments.embedding_model
    if changes:
        project = project.with_settings(**changes)
        project.save()

    Corpus.load(project.paths.sources).register(arguments.folder.resolve())
    config = project.settings_for(Settings.from_env())
    embedder = build_embedding_provider(
        project.settings.embedding_provider,
        config,
        model=project.settings.embedding_model or None,
    )
    report = index_project(project, embedder, config=config)

    print(report.summary)
    for result in report.failed:
        print(f'{result.ref}: {result.problem}')


if __name__ == '__main__':
    main()
