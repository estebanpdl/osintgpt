"""Build embedding and generation backends without a project, by id or OpenAI-compatible URL.

Usage:
    python examples/library/custom_provider.py TEXT \
        --embedding-provider PROVIDER --generation-provider PROVIDER [--*-model MODEL]
    python examples/library/custom_provider.py TEXT --base-url URL \
        --embedding-model MODEL --generation-model MODEL [--api-key-env VAR]
"""

import argparse
import os

from osintgpt import Settings
from osintgpt.llm import (
    EMBEDDING_BACKENDS,
    GENERATION_BACKENDS,
    OpenAICompatEmbedding,
    OpenAICompatGeneration,
    UsageRecorder,
    audit_locality,
    build_embedding_provider,
    build_generation_provider,
)
from osintgpt.llm.embedding_retry import RetryPolicy


def registered(arguments, settings: Settings, recorder: UsageRecorder):
    """Backends osintgpt knows by id; credentials come from ``settings``."""
    embedder = build_embedding_provider(
        arguments.embedding_provider, settings,
        model=arguments.embedding_model, recorder=recorder,
    )
    generator = build_generation_provider(
        arguments.generation_provider, settings,
        model=arguments.generation_model, recorder=recorder,
    )
    report = audit_locality(
        settings, arguments.embedding_provider, arguments.generation_provider,
        embedder.model, generator.model,
    )
    print(f'Locality: {report.summary}')

    return embedder, generator


def compatible_endpoint(arguments, recorder: UsageRecorder):
    """Any server speaking the OpenAI API, such as vLLM, LM Studio, or a gateway."""
    if not (arguments.embedding_model and arguments.generation_model):
        raise SystemExit('--base-url needs --embedding-model and --generation-model')

    # Keyless local servers still need a non-empty placeholder.
    api_key = os.environ.get(arguments.api_key_env, '') or 'unused'
    embedder = OpenAICompatEmbedding(
        model=arguments.embedding_model, api_key=api_key,
        base_url=arguments.base_url, billable=arguments.billable,
        provider='custom', recorder=recorder,
        retry_policy=RetryPolicy(max_attempts=arguments.max_attempts),
    )
    generator = OpenAICompatGeneration(
        model=arguments.generation_model, api_key=api_key,
        base_url=arguments.base_url, billable=arguments.billable,
        provider='custom', recorder=recorder,
    )
    print(f'Endpoint: {arguments.base_url} (billable={arguments.billable})')

    return embedder, generator


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Construct providers directly and make one call to each.'
    )
    parser.add_argument('text', help='text to embed and summarize')
    parser.add_argument('--embedding-provider', choices=list(EMBEDDING_BACKENDS))
    parser.add_argument('--generation-provider', choices=list(GENERATION_BACKENDS))
    parser.add_argument('--embedding-model', help='model name')
    parser.add_argument('--generation-model', help='model name')
    parser.add_argument('--base-url', help='OpenAI-compatible endpoint instead of a registered id')
    parser.add_argument('--api-key-env', default='OPENAI_COMPAT_API_KEY',
                        help='environment variable holding the endpoint key')
    parser.add_argument('--billable', action='store_true', help='the endpoint charges per call')
    parser.add_argument('--max-attempts', type=int, default=6,
                        help='embedding attempts per batch; 1 disables retries')
    parser.add_argument('--env-file', help='.env file to read credentials from')
    arguments = parser.parse_args()

    recorder = UsageRecorder()
    if arguments.base_url:
        embedder, generator = compatible_endpoint(arguments, recorder)
    elif arguments.embedding_provider and arguments.generation_provider:
        settings = Settings.from_env(arguments.env_file)
        embedder, generator = registered(arguments, settings, recorder)
    else:
        raise SystemExit('give --base-url, or both --embedding-provider and --generation-provider')

    vector = embedder.embed([arguments.text])[0]
    print(f'{embedder.model}: {len(vector)} dimensions')
    print(generator.generate('Summarize the text in one sentence.', arguments.text))
    print(f'Usage: {recorder.summary}')


if __name__ == '__main__':
    main()
