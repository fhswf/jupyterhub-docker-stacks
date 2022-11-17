# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Ubuntu 22.04 (jammy)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=jammy
ARG ROOT_CONTAINER=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04
FROM ubuntu as BUILDBEGIN

ARG ROOT_CONTAINER=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu18.04
ENV ROOT_CONTAINER=${ROOT_CONTAINER:-void}
# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "hello build ${ENV}"

RUN echo "${ROOT_CONTAINER}"

LABEL mylabel="hello"
RUN echo "${ROOT_CONTAINER:7:11}"

RUN if [[ ${ROOT_CONTAINER:7:11} == cuda* ]] ; then echo "has cuda" ; else echo "no cuda" ; fi

FROM BUILDBEGIN-${ROOT_CONTAINER} as CONDITION
LABEL fhswf.jupyterhub.runtime="NVIDIA-GPU"
