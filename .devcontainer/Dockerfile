# syntax=docker/dockerfile:1.4

FROM peaceiris/hugo:v0.120.4

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Install required system packages
RUN printf "deb http://deb.debian.org/debian bullseye-backports main" > /etc/apt/sources.list.d/backports_git.list && \
    apt-get update \
    && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    direnv \
    git/bullseye-backports \
    ssh-client \
    expect \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 

# This cache directory will mounted as a volume inside the container to preserve
# it across rebuilds. Docker mounts "take" ownership and permissions from
# existing folders, so we create & chown ahead of time.
RUN mkdir /home/vscode/.cache && chown vscode:vscode /home/vscode/.cache
RUN pip3 install pre-commit

USER $USERNAME

# Set bash as default shell
ENV SHELL=/bin/bash

ENTRYPOINT []
