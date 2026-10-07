"""Search several isolated projects and keep each result's project identity.

Usage:
    python examples/library/across_projects.py QUERY PROJECT [PROJECT ...] \
        [--embedding-provider PROVIDER --embedding-model MODEL]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, search_across_projects
from osintgpt.llm import build_embedding_provider
from osintgpt.vector_store import store_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Search compatible projects as one ranked result set.'
    )
    parser.add_argument('query', help='query to embed')
    parser.add_argument('projects', type=Path, nargs='+', help='project directories')
    parser.add_argument('--embedding-provider', help='provider id (default: first project)')
    parser.add_argument('--embedding-model', help='model name (default: first project)')
    parser.add_argument('--top-k', type=int, default=10, help='merged result limit')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    projects = [Project.load(path) for path in arguments.projects]
    first = projects[0].settings
    settings = Settings.from_env(arguments.env_file)
    embedder = build_embedding_provider(
        arguments.embedding_provider or first.embedding_provider,
        settings,
        model=arguments.embedding_model or first.embedding_model or None,
    )

    def configured_store(project):
        return store_for(project, project.settings_for(settings))

    # Projects indexed with another model are excluded and named in the notice.
    results = search_across_projects(
        projects,
        arguments.query,
        embedder,
        top_k=arguments.top_k,
        store_factory=configured_store,
    )

    if results.notice:
        print(results.notice)
    for rank, hit in enumerate(results, 1):
        citation = hit.payload.chunk.citation
        print(f'{rank}. {hit.score:.3f}  {hit.project_slug}  {citation}')


if __name__ == '__main__':
    main()
