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
        # Ensure ssh-agent is running on host
        if [ -z "${SSH_AUTH_SOCK:-}" ]; then
            eval "$(ssh-agent -s)" > /dev/null 2>&1 || true
        fi
        # Add default host SSH keys if no identities are loaded
        if ! ssh-add -l > /dev/null 2>&1; then
            for key in ~/.ssh/id_ed25519 ~/.ssh/id_rsa ~/.ssh/id_ecdsa; do
                if [ -f "$key" ]; then
                    ssh-add "$key" 2> /dev/null || true
                fi
            done
        fi
    fi
fi

# Pre-create host directories for AI agents to ensure bind mounts succeed
mkdir -p "${HOME}/.claude" "${HOME}/.gemini"

