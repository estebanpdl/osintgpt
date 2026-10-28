"""Construct modern providers after removal of the legacy API.

Usage:
    python examples/library/migrating_from_0_1.py ROLE PROVIDER MODEL
"""

import argparse

from osintgpt import Settings
from osintgpt.llm import build_embedding_provider, build_generation_provider


# Legacy calls are shown only as comments. The next minor after 0.7.1 removes
# them, changing the earlier 1.0 promise. See docs/migration-legacy-api.md for
# behavior/storage differences, SQL export, and the osintgpt<0.8 fallback.
#
# from osintgpt.embeddings import OpenAIEmbeddingGenerator
# old_embedder = OpenAIEmbeddingGenerator('.env')
# new_embedder = build_embedding_provider('openai', Settings.from_env())
#
# from osintgpt.llms import OpenAIGPT
# old_generator = OpenAIGPT('.env')
# new_generator = build_generation_provider(
#     'openai', Settings.from_env(), model=MODEL
# )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Construct a provider through the supported factory API.'
    )
    parser.add_argument('role', choices=('embedding', 'generation'))
    parser.add_argument('provider', help='provider id')
    parser.add_argument('model', help='model name')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    settings = Settings.from_env(arguments.env_file)
    factory = (
        build_embedding_provider
        if arguments.role == 'embedding'
        else build_generation_provider
    )
    provider = factory(arguments.provider, settings, model=arguments.model)
    print(f'{type(provider).__name__}: {provider.model}')


if __name__ == '__main__':
    main()
