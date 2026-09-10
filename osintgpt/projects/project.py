# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: project.py
# Description: A project is a directory holding one case: its own settings, its
#   own store, its own corpus registry. Nothing is shared between projects.
# =================================================================================

# import modules
import re
import uuid

# import submodules
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

# type hints
from typing import Optional, Union

# import osintgpt config
from osintgpt.config import Settings

from .paths import ProjectPaths, default_home
from .settings import ProjectSettings
from .toml_io import read_toml, write_toml

# Written at the top of every project.toml. A project directory is the thing
# people zip up and hand to a colleague, so the warning belongs where someone
# about to paste a key will read it.
CONFIG_HEADER = '''\
# osintgpt project
#
# Do not put API keys in this file. Secrets belong in your environment or your
# user config; this directory is meant to be copied, archived and shared.

'''

# turn a display name into a directory-safe slug
def slugify(name: str) -> str:
    '''
    A lowercase, hyphenated form of a name, safe as a directory name.

    Args:
        name (str): Display name.

    Returns:
        str: The slug, or 'project' when nothing usable survives.
    '''
    slug = re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

    return slug or 'project'


# Project class
@dataclass(frozen=True)
class Project:
    '''
    One case: identity, settings, and the paths they live in.
    '''
    id: str
    slug: str
    name: str
    created_at: str
    paths: ProjectPaths
    settings: ProjectSettings = ProjectSettings()

    # create a new project on disk
    @classmethod
    def create(
        cls,
        name: str,
        home: Optional[Union[str, Path]] = None,
        path: Optional[Union[str, Path]] = None,
        settings: Optional[ProjectSettings] = None
    ):
        '''
        Create a project directory and write its configuration.

        Args:
            name (str): Display name. The slug is derived from it.
            home (Union[str, Path], optional): osintgpt home to create the \
                project under. Defaults to `default_home()`.
            path (Union[str, Path], optional): An explicit directory, for a \
                project that belongs on a case-specific or encrypted volume. \
                Wins over `home`.
            settings (ProjectSettings, optional): Initial settings.

        Raises:
            FileExistsError: If the target directory already holds a project.

        Returns:
            Project: The created project.
        '''
        slug = slugify(name)
        paths = (
            ProjectPaths(Path(path)) if path is not None
            else ProjectPaths.under_home(
                Path(home) if home is not None else default_home(), slug
            )
        )

        if paths.config.exists():
            raise FileExistsError(
                f'a project already exists at {paths.root}'
            )

        project = cls(
            id=uuid.uuid4().hex,
            slug=slug,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
            paths=paths,
            settings=settings or ProjectSettings()
        )
        paths.create_directories()
        project.save()

        return project

    # load an existing project
    @classmethod
    def load(cls, path: Union[str, Path]):
        '''
        Read a project from its directory.

        Args:
            path (Union[str, Path]): The project root.

        Raises:
            FileNotFoundError: If the directory holds no project.toml.

        Returns:
            Project: The project as configured on disk.
        '''
        paths = ProjectPaths(Path(path))
        if not paths.config.is_file():
            raise FileNotFoundError(f'no project at {paths.root}')

        data = read_toml(paths.config)
        identity = data.get('project', {})

        return cls(
            id=identity.get('id', ''),
            slug=identity.get('slug', paths.root.name),
            name=identity.get('name', paths.root.name),
            created_at=identity.get('created_at', ''),
            paths=paths,
            settings=ProjectSettings.from_dict(data.get('settings', {}))
        )

    # write configuration back to disk
    def save(self) -> None:
        '''
        Write project.toml, replacing whatever was there.
        '''
        document = {
            'project': {
                'id': self.id,
                'slug': self.slug,
                'name': self.name,
                'created_at': self.created_at
            },
            'settings': self.settings.to_dict()
        }
        write_toml(self.paths.config, document, header=CONFIG_HEADER)

    # replace settings
    def with_settings(self, **changes):
        '''
        A copy carrying changed settings. Does not write to disk.

        Args:
            **changes: ProjectSettings fields to replace.

        Returns:
            Project: A new instance; the original is unchanged.
        '''
        return replace(self, settings=replace(self.settings, **changes))

    # project choices with user defaults filling the gaps
    def effective_settings(
        self, defaults: Optional[ProjectSettings] = None
    ) -> ProjectSettings:
        '''
        This project's settings, with anything it left unset taken from user
        defaults.

        Args:
            defaults (ProjectSettings, optional): User defaults, loaded by the \
                caller from a home it names.

        Returns:
            ProjectSettings: The settings this project actually runs with.
        '''
        if defaults is None:
            return self.settings

        # False and 0 are choices; only '' and None mean "not chosen here".
        filled = {
            field.name: getattr(defaults, field.name)
            for field in fields(ProjectSettings)
            if getattr(self.settings, field.name) in ('', None)
            and getattr(defaults, field.name) not in ('', None)
        }

        return replace(self.settings, **filled)

    # compose project choices onto caller-supplied configuration
    def settings_for(
        self, base: Settings, defaults: Optional[ProjectSettings] = None
    ) -> Settings:
        '''
        Apply this project's choices to configuration the caller supplies.

        Resolution runs explicit argument, then project, then user defaults,
        then library default: a value set on `base` wins, the project fills
        what `base` left unset, and `defaults` fills what neither chose. The
        conversation store is always redirected into the project, so two
        projects cannot write to one log.

        Args:
            base (Settings): Configuration carrying credentials. Secrets are \
                never read from the project.
            defaults (ProjectSettings, optional): User defaults, loaded by the \
                caller from a home it names. Nothing is discovered here.

        Returns:
            Settings: A copy with the project's choices applied.
        '''
        settings = self.effective_settings(defaults)
        overrides = {'sql_db_file_path': str(self.paths.store)}

        for field, target in (
            ('generation_model', 'openai_gpt_model'),
            ('embedding_model', 'openai_embedding_model')
        ):
            if not getattr(base, target) and getattr(settings, field):
                overrides[target] = getattr(settings, field)

        return base.with_overrides(**overrides)
