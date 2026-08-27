# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: connection.py
# Description: Opening a Qdrant connection from settings. Shared so the two
#   classes that need one cannot drift apart on how a server is reached.
# =================================================================================

# import modules <Qdrant>
import qdrant_client

# type hints
from typing import Tuple

# import osintgpt config
from osintgpt.config import Settings

# import exceptions
from osintgpt.exceptions.errors import MissingEnvironmentVariableError

REMOTE = 'remote'
LOCAL = 'local'

# The client defaults to five seconds, which is tuned for search. Creating a
# collection takes longer than that on a real server, so the first write of a
# project would time out — measured, not guessed.
TIMEOUT_SECONDS = 60

UNREACHABLE = (
    'Unable to establish a connection to the Qdrant server. Please ensure '
    'that the Qdrant server is up and running. If you are using this '
    'locally, make sure to start the Qdrant server before using this feature.'
)


# open a connection from settings
def connect(settings: Settings) -> Tuple['qdrant_client.QdrantClient', str]:
    '''
    Build a client and confirm the server answers.

    A remote pair (api key + url) wins over a local pair (host + port) when
    both are present, so a project pointed at a hosted cluster is not silently
    served by a container someone left running.

    Args:
        settings (Settings): Configuration carrying the Qdrant fields.

    Raises:
        MissingEnvironmentVariableError: If neither pair is complete.
        ConnectionError: If the server does not answer.

    Returns:
        Tuple[QdrantClient, str]: The client, and which pair was used.
    '''
    use_remote = settings.qdrant_api_key and settings.qdrant_url
    use_local = settings.qdrant_port and settings.qdrant_host

    if not (use_remote or use_local):
        raise MissingEnvironmentVariableError(
            'QDRANT_API_KEY or QDRANT_URL or QDRANT_HOST or QDRANT_PORT',
            hint='a remote Qdrant needs an api key and a url; a local one '
                 'needs a host and a port'
        )

    if use_remote:
        client = qdrant_client.QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            https=True,
            timeout=TIMEOUT_SECONDS
        )
        kind = REMOTE
    else:
        client = qdrant_client.QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=TIMEOUT_SECONDS
        )
        kind = LOCAL

    # A client constructs without touching the network, so an unreachable
    # server would otherwise surface at the first search rather than here.
    try:
        client.get_collections()
    except Exception:
        raise ConnectionError(UNREACHABLE) from None

    return client, kind
