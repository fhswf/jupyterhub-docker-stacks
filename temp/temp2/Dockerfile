# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Ubuntu 22.04 (jammy)
# https://hub.docker.com/_/ubuntu/?tab=tags&name=jammy
FROM kuhmichel_devimage

LABEL mylabel="hello"

RUN echo ${cuda}

RUN if [[ ${cuda} == 1]] ; then echo "$ROOT_CONTAINER stats with nvida" ; else echo "$ROOT_CONTAINER starts with something other" ; fi
