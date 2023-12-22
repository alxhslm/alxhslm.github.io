#!/bin/bash

# Unofficial Bash Strict Mode
set -euo pipefail
IFS=$'\n\t'

WORKSPACE="/workspace"

# Copy bash/zsh history from persistent cache
touch /home/vscode/.cache/.bash_history /home/vscode/.cache/.zsh_history
ln -Fs /home/vscode/.cache/.bash_history /home/vscode/.bash_history
ln -Fs /home/vscode/.cache/.zsh_history /home/vscode/.zsh_history

# Start in a well-know location
cd $WORKSPACE

# Install pre-commit hooks
if [ ! -e ".git/hooks/pre-commit" ] ; then
  pre-commit install
fi

# Install direnv hooks
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc
direnv allow
eval "$(direnv export bash)"

echo '
Dev container is ready to use!
'
