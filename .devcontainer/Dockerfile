FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

# VARIANT can be either 'hugo' for the standard version or 'hugo_extended' for the extended version.
ARG VARIANT=hugo
# VERSION can be either 'latest' or a specific version number
ARG VERSION=latest

# Install required system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    direnv \
    git \
    curl \
    expect \
    && rm -rf /var/lib/apt/lists/*

# Download Hugo
COPY scripts/*.sh /tmp/scripts/
RUN /tmp/scripts/install_hugo.sh ${VARIANT} ${VERSION}

# Hugo dev server port
EXPOSE 1313

# Install pre-commit
RUN pip install pre-commit

# This cache directory will mounted as a volume inside the container to preserve
# it across rebuilds. Docker mounts "take" ownership and permissions from
# existing folders, so we create & chown ahead of time.
RUN mkdir /home/vscode/.cache && chown vscode:vscode /home/vscode/.cache


# [Optional] Uncomment if you want to install more global node packages
# RUN sudo -u node npm install -g <your-package-list-here>
