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

# instal torch with or without cuda dependen on the base image used
# TODO with ${ROOT_CONTAINER:7:11} we can read out the cuda version and use it for the repo version
#  the problem: torch takes a while to get the latest updates and is most liky one minor version behind
RUN if [[ ${ROOT_CONTAINER:7:11} == cuda* ]] ; then  pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 && pip cache remove * ; else  pip install torch torchvision torchaudio && pip cache remove * ; fi