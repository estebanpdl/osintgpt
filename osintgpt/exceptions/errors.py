# -*- coding: utf-8 -*-

# ========================================================================
# osintgpt
#
# Author: @estebanpdl
#
# File: errors.py
# Description: This file contains the custom error classes for osintgpt.
# ========================================================================

# MissingEnvironmentVariableError class
class MissingEnvironmentVariableError(Exception):
    '''
    MissingEnvironmentVariableError class

    This class is a custom error class for missing environment variables.
    '''
    def __init__(self, variable_name):
        '''
        '''
        message = f"Missing required environment variable: {variable_name}"
        super().__init__(message)
