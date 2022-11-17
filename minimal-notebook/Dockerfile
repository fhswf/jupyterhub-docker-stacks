# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG REGISTRY=ghcr.io 
ARG OWNER=fhswf/jupyterhub-docker-stacks
ARG BASE_CONTAINER=$REGISTRY/$OWNER/base-notebook
FROM $BASE_CONTAINER

LABEL maintainer="FH-SWF <https://github.com/fhswf/jupyterhub-docker-stacks>, forked from Jupyter Project <jupyter@googlegroups.com>"

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for fully functional notebook server
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    # Common useful utilities
    git \
    nano-tiny \
    tzdata \
    unzip \
    vim-tiny \
    # git-over-ssh
    openssh-client \
    # less is needed to run help in R
    # see: https://github.com/jupyter/docker-stacks/issues/1588
    less \
    # nbconvert dependencies
    # https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic && \
    apt-get clean && rm -rf /var/lib/apt/lists/* \
    fonts-dejavu \
    tzdata \
    gfortran \
    gcc \
    htop \
    wget \
    iputils-ping \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create alternative for nano -> nano-tiny
RUN update-alternatives --install /usr/bin/nano nano /bin/nano-tiny 10

# set LD_path for cuda versions as it is needed by some libraries, 
#  does no harm when cuda it self is not installed
ENV LD_LIBRARY_PATH=${CONDA_DIR}/lib/:${LD_LIBRARY_PATH}
RUN echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment 
#  same for CPATH (used for compilation)
ENV CPATH=$CONDA_DIR/include:$CPATH
RUN echo CPATH="${CPATH}" >> /etc/environment

# Switch back to jovyan to avoid accidental container runs as root
USER ${NB_UID}

# to pull content to the images, out setup make great us if this
RUN pip install nbgitpuller && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
