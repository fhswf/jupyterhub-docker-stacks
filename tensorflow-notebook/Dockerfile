# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG REGISTRY=ghcr.io 
ARG OWNER=fhswf/jupyterhub-docker-stacks
ARG PARENT=scipy-notebook
ARG BASE_CONTAINER=$REGISTRY/$OWNER/$PARENT
FROM $BASE_CONTAINER

LABEL maintainer="FH-SWF <https://github.com/fhswf/jupyterhub-docker-stacks>, forked from Jupyter Project <jupyter@googlegroups.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Tensorflow
RUN mamba install --quiet --yes \
    'cudatoolkit' 'tensorflow' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
