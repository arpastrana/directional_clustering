"""
********************************************************************************
directional_clustering
********************************************************************************

.. currentmodule:: directional_clustering


.. toctree::
    :maxdepth: 1

    directional_clustering.clustering
    directional_clustering.fields
    directional_clustering.mesh
    directional_clustering.plotters
    directional_clustering.transformations
"""

from __future__ import print_function

import os
import sys


__copyright__ = "Princeton University"
__license__ = "MIT License"
__version__ = "0.1.0"


HERE = os.path.dirname(__file__)

HOME = os.path.abspath(os.path.join(HERE, "../../"))
DATA = os.path.abspath(os.path.join(HOME, "data"))
DOCS = os.path.abspath(os.path.join(HOME, "docs"))
FIELDS = os.path.abspath(os.path.join(HOME, 'data/rawfield/'))
JSON = os.path.abspath(os.path.join(HOME, 'data/json/'))
OFF = os.path.abspath(os.path.join(HOME, 'data/json/'))
SCRIPTS = os.path.abspath(os.path.join(HOME, 'scripts'))
TEMP = os.path.abspath(os.path.join(HOME, "temp"))
TESTS = os.path.abspath(os.path.join(HOME, 'tests'))

# Check if package is installed from git
# If that's the case, try to append the current head's hash to __version__
try:
    git_head_file = compas._os.absjoin(HOME, '.git', 'HEAD')

    if os.path.exists(git_head_file):
        # git head file contains one line that looks like this:
        # ref: refs/heads/master
        with open(git_head_file, 'r') as git_head:
            _, ref_path = git_head.read().strip().split(' ')
            ref_path = ref_path.split('/')

            git_head_refs_file = compas._os.absjoin(HOME, '.git', *ref_path)

        if os.path.exists(git_head_refs_file):
            with open(git_head_refs_file, 'r') as git_head_ref:
                git_commit = git_head_ref.read().strip()
                __version__ += '-' + git_commit[:8]
except Exception:
    pass

__all__ = ["HOME", "DATA", "DOCS", "TEMP", "TESTS"]
