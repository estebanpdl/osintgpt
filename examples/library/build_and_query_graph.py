"""Build a project's sourced graph, then inspect a neighborhood or path.

Usage:
    python examples/library/build_and_query_graph.py PROJECT ENTITY [options]
"""

import argparse
from pathlib import Path

from osintgpt import Project, Settings, build_graph, graph_for
from osintgpt.graph import neighbors, path_between
from osintgpt.llm import build_generation_provider


def print_edge(edge, depth: int | None = None) -> None:
    """Print one relationship with the evidence that supports it."""
    prefix = f'depth {depth}: ' if depth is not None else ''
    print(f'{prefix}{edge.source} --{edge.relation}--> {edge.target}')
    print(f'  {edge.ref}: {edge.evidence}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build and traverse the evidence-carrying project graph.'
    )
    parser.add_argument('project', type=Path, help='project directory')
    parser.add_argument('entity', help='entity at which to start')
    parser.add_argument('--target', help='find a shortest path to this entity')
    parser.add_argument('--generation-provider', help='provider id')
    parser.add_argument('--generation-model', help='model name')
    parser.add_argument('--incremental', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    arguments = parser.parse_args()

    project = Project.load(arguments.project)
    config = project.settings_for(Settings.from_env())
    provider = arguments.generation_provider or project.settings.generation_provider
    model = arguments.generation_model or project.settings.generation_model or None
    generator = build_generation_provider(provider, config, model=model)
    report = build_graph(
        project,
        generator,
        incremental=arguments.incremental,
        rebuild=arguments.rebuild,
    )
    print(report.summary)
    if report.refused:
        return

    with graph_for(project) as graph:
        if arguments.target:
            path = path_between(graph, arguments.entity, arguments.target)
            if path is None:
                print('No path found.')
                return
            for edge in path.edges:
                print_edge(edge)
            return

        for hit in neighbors(graph, arguments.entity):
            print_edge(hit.edge, hit.depth)


if __name__ == '__main__':
    main()
