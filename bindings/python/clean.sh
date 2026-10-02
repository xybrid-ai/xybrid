#!/usr/bin/env bash
# Python binding: setuptools output, the test cache, and the natives
# build-python-bolt.sh stages into xybrid/_bolt.
. "$(dirname "$0")/../../tools/scripts/clean-lib.sh"
clean_paths build dist '*.egg-info' .pytest_cache
clean_paths 'xybrid/_bolt/*.dylib' 'xybrid/_bolt/*.so'
clean_paths 'xybrid/_bolt/*.dll' 'xybrid/_bolt/*.pyd'
clean_run "$@"
