"""Build embedding and generation backends without creating a project.

Usage:
    python examples/library/custom_provider.py TEXT SYSTEM PROMPT \
        --embedding-provider PROVIDER --embedding-model MODEL \
        --generation-provider PROVIDER --generation-model MODEL
"""

import argparse

from osintgpt import Settings
from osintgpt.llm import build_embedding_provider, build_generation_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Use provider factories independently of a project.'
    )
    parser.add_argument('text', help='text to embed')
    parser.add_argument('system', help='system instruction')
    parser.add_argument('prompt', help='user message')
    parser.add_argument('--embedding-provider', required=True, help='provider id')
    parser.add_argument('--embedding-model', required=True, help='model name')
    parser.add_argument('--generation-provider', required=True, help='provider id')
    parser.add_argument('--generation-model', required=True, help='model name')
    arguments = parser.parse_args()

    settings = Settings.from_env()
    embedder = build_embedding_provider(
        arguments.embedding_provider,
        settings,
        model=arguments.embedding_model,
    )
    generator = build_generation_provider(
        arguments.generation_provider,
        settings,
        model=arguments.generation_model,
    )

    vector = embedder.embed([arguments.text])[0]
    print(f'{embedder.model}: {len(vector)} dimensions')
    print(generator.generate(arguments.system, arguments.prompt))


if __name__ == '__main__':
    main()
