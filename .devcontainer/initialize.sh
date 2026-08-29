#!/bin/bash
# This script will be run on the *host* machine before VS Code starts building
# devcontainer image. Use it to prepare anything necessary for building the dev
# container or to check if the host environment meets requirements.

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

ENV_FILE=".env"
ENV_FILE_TEMPLATE=".env.default"

# Create .env file if necessary
if [ ! -e $ENV_FILE ] ; then
  cp $ENV_FILE_TEMPLATE $ENV_FILE
fi

if [ -z "${CODESPACES:+x}" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Macs only add keys to agent on demand (no-op if already loaded)
        ssh-add --apple-load-keychain 2> /dev/null || true
    elif [[ "$OSTYPE" == "linux"* ]]; then
        mkdir -p "${HOME}/.ssh"
        # Ensure ssh-agent is running at fixed socket location ~/.ssh/agent.sock
        if [ ! -S "${HOME}/.ssh/agent.sock" ]; then
            rm -f "${HOME}/.ssh/agent.sock"
            eval "$(ssh-agent -a "${HOME}/.ssh/agent.sock")" > /dev/null 2>&1 || true
        fi
        export SSH_AUTH_SOCK="${HOME}/.ssh/agent.sock"
        ssh-add 2> /dev/null || true
    fi
fi

# Pre-create host directories for AI agents and GitHub CLI to ensure bind mounts succeed
mkdir -p "${HOME}/.claude" "${HOME}/.gemini" "${HOME}/.config/gh"

