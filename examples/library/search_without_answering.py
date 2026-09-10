"""Compare semantic retrieval with rank-fused semantic and exact retrieval.

Usage:
    python examples/library/search_without_answering.py PROJECT QUERY \
        --term EXACT_TERM [--term ANOTHER_TERM]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, hybrid_search, search_project
from osintgpt.llm import build_embedding_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print semantic results beside fused retrieval results.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('query', help='semantic query')
    parser.add_argument(
        '--term', action='append', default=[], help='exact term; repeatable'
    )
    parser.add_argument('--top-k', type=int, default=10, help='results per list')
    parser.add_argument('--embedding-provider', help='provider id')
    parser.add_argument('--embedding-model', help='model name')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env())
    provider = arguments.embedding_provider or project.settings.embedding_provider
    model = arguments.embedding_model or project.settings.embedding_model or None
    embedder = build_embedding_provider(provider, config, model=model)

    semantic = search_project(
        project, arguments.query, embedder, top_k=arguments.top_k
    )
    fused = hybrid_search(
        project,
        arguments.query,
        embedder,
        terms=arguments.term,
        top_k=arguments.top_k,
    )

    print('Semantic')
    for rank, result in enumerate(semantic, 1):
        print(f'{rank}. {result.score:.3f}  {result.chunk.citation}')

    print('\nFused')
    for rank, result in enumerate(fused, 1):
        legs = ', '.join(
            f'{name}:{position}' for name, position in result.ranks.items()
        )
        print(f'{rank}. {result.score:.4f}  {result.result.chunk.citation}  {legs}')


if __name__ == '__main__':
    main()
