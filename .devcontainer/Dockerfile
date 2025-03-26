# [Choice] Python version (use -bullseye variants on local arm64/Apple Silicon): 3, 3.10, 3.9, 3.8, 3.7, 3.6, 3-bullseye, 3.10-bullseye, 3.9-bullseye, 3.8-bullseye, 3.7-bullseye, 3.6-bullseye, 3-buster, 3.10-buster, 3.9-buster, 3.8-buster, 3.7-buster, 3.6-buster
ARG VARIANT="3.10-bullseye"
FROM mcr.microsoft.com/devcontainers/python:1-${VARIANT}

# System Updates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Update pip
RUN pip install --upgrade pip

#Install uv MY way
USER vscode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH = "${PATH}:/home/vscode/.local/bin"

## Git setup
RUN git config --global diff.tool vscode && git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
