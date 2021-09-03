import os
import sys

import pkg_resources
from autotabular.__version__ import __version__  # noqa (imported but unused)
from autotabular.util import dependencies

requirements = pkg_resources.resource_string('autotabular', 'requirements.txt')
requirements = requirements.decode('utf-8')

dependencies.verify_packages(requirements)

if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: https://automl.github.io'
        '/auto-sklearn/stable/installation.html#windows-osx-compability' %
        sys.platform)

if sys.version_info < (3, 6):
    raise ValueError(
        'Unsupported python version %s found. Auto-sklearn requires Python '
        '3.6 or higher.' % sys.version_info)
