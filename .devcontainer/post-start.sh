#!/bin/bash
# This script is executed every time the dev container starts

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

# Make sure Git sees workspace as safe
#
# This can't be run in post-create.sh as it would prevent VS Code from copying
# user's .gitconfig and would decrease Dev Container's personalisation.
#
# Note: According to the docs, this shouldn't be necessary, as /workspace and
# everything below is owned by `vscode` user, but somehow git has problems with
# this dev container setup. Docs:
# https://git-scm.com/docs/git-config/2.35.2#Documentation/git-config.txt-safedirectory
if [[ ! `git config --global --get-all safe.directory | grep '^/workspace$'` ]]; then
  # Adding only when not added already
  git config --global --add safe.directory /workspace
fi

# Ensure /workspace maps to alxhslm-github-io in ~/.gemini/projects.json for shared AGY conversation history
PROJECTS_FILE="/home/vscode/.gemini/projects.json"
if [ -f "$PROJECTS_FILE" ]; then
  python3 -c "
import json
p = '$PROJECTS_FILE'
try:
    with open(p, 'r') as f:
        data = json.load(f)
    if 'projects' in data:
        data['projects']['/workspace'] = 'alxhslm-github-io'
        with open(p, 'w') as f:
            json.dump(data, f, indent=2)
except Exception:
    pass
" || true
fi

