# -*- coding: utf-8 -*-

# =================================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: sitecustomize.py
# Description: Refuses outbound network connections. Python imports this
#   automatically when `ci/` is on PYTHONPATH, so the CI offline job proves the
#   suite never reaches a provider rather than trusting that it doesn't.
# =================================================================================

# import modules
import socket

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_create_connection = socket.create_connection

# Loopback stays open: a test that needs a local server is testing something
# real, while a route off the machine is a provider call that escaped its stub.
LOOPBACK = ('127.0.0.1', '::1', 'localhost', '')


class NetworkBlocked(RuntimeError):
    pass


def _is_local(address):
    if not isinstance(address, tuple) or not address:
        return True

    return address[0] in LOOPBACK


def _guard(original):
    def wrapper(self, address, *args, **kwargs):
        if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
            raise NetworkBlocked(
                f'outbound connection to {address} blocked — a test reached the '
                'network instead of a stub'
            )

        return original(self, address, *args, **kwargs)

    return wrapper


def _guard_create_connection(address, *args, **kwargs):
    if not _is_local(address):
        raise NetworkBlocked(
            f'outbound connection to {address} blocked — a test reached the '
            'network instead of a stub'
        )

    return _real_create_connection(address, *args, **kwargs)


socket.socket.connect = _guard(_real_connect)
socket.socket.connect_ex = _guard(_real_connect_ex)
socket.create_connection = _guard_create_connection
