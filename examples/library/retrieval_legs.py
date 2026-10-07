"""Run the semantic and lexical legs separately, then fuse them by rank.

Usage:
    python examples/library/retrieval_legs.py PROJECT QUERY \
        [--term EXACT_TERM ...] [--derive-terms] [--lexical-weight W]
"""

import argparse
from pathlib import Path

from osintgpt import (
    Project,
    Settings,
    derive_search_terms,
    lexical_search,
    reciprocal_rank_fusion,
    search_project,
)
from osintgpt.llm import UsageRecorder, build_embedding_provider, build_generation_provider
from osintgpt.vector_store import store_for


def print_leg(name: str, results) -> None:
    print(f'\n{name}')
    if not results:
        print('  (no results)')
    for rank, result in enumerate(results, 1):
        terms = f'  terms={list(result.terms)}' if result.terms else ''
        print(f'  {rank}. {result.score:.3f}  {result.chunk.citation}{terms}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare each retrieval leg with their rank fusion.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('query', help='semantic query')
    parser.add_argument('--term', action='append', default=[], help='exact term; repeatable')
    parser.add_argument('--derive-terms', action='store_true',
                        help='ask the generation model for terms (one call)')
    parser.add_argument('--lexical-weight', type=float, default=1.0,
                        help='fusion multiplier for the lexical leg')
    parser.add_argument('--top-k', type=int, default=5, help='results per list')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env(arguments.env_file))
    recorder = UsageRecorder()
    embedder = build_embedding_provider(
        project.settings.embedding_provider,
        config,
        model=project.settings.embedding_model or None,
        recorder=recorder,
    )

    terms = list(arguments.term)
    if arguments.derive_terms:
        generator = build_generation_provider(
            project.settings.generation_provider,
            config,
            model=project.settings.generation_model or None,
            recorder=recorder,
        )
        terms += derive_search_terms(generator, arguments.query)
        print(f'terms: {terms}')

    # One store for every leg; remote backends need the connection settings.
    store = store_for(project, config)
    try:
        semantic = search_project(
            project, arguments.query, embedder, top_k=arguments.top_k, store=store
        )
        lexical = lexical_search(
            project, terms, embedding_model=embedder.model, store=store
        )[:arguments.top_k]
    finally:
        if hasattr(store, 'close'):
            store.close()

    print_leg('Semantic (cosine similarity)', semantic)
    print_leg('Lexical (share of terms matched)', lexical)

    fused = reciprocal_rank_fusion(
        {'semantic': semantic, 'lexical': lexical},
        limit=arguments.top_k,
        weights={'lexical': arguments.lexical_weight},
    )
    print('\nFused (reciprocal rank)')
    for rank, result in enumerate(fused, 1):
        legs = ', '.join(f'{leg} #{position}' for leg, position in result.ranks.items())
        print(f'  {rank}. {result.score:.4f}  {result.result.chunk.citation}  [{legs}]')

    print(f'\nUsage: {recorder.summary}')


if __name__ == '__main__':
    main()
