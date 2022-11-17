#!/usr/bin/env python3
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
import argparse
import logging
from pathlib import Path
from platform import platform

from tagging.docker_runner import DockerRunner
from tagging.get_platform import get_platform
from tagging.get_taggers_and_manifests import get_taggers_and_manifests

LOGGER = logging.getLogger(__name__)


def write_tags_file(
    short_image_name: str,
    tag_prefix: str, 
    owner: str,
    tags_dir: Path,
    registry: str
) -> None:
    """
    Writes tags file for the image <registry>/<owner>/<short_image_name>:latest
    """
    LOGGER.info(f"Tagging image: {short_image_name}")
    taggers, _ = get_taggers_and_manifests(short_image_name)

    image = f"{registry}/{owner}/{short_image_name}:latest"
    platform = get_platform()
    filename = f"{platform}-{tag_prefix}-{short_image_name}.txt"

    tags = [f"{registry}/{owner}/{short_image_name}:{platform}-{tag_prefix}-latest"]
    with DockerRunner(image) as container:
        for tagger in taggers:
            tagger_name = tagger.__class__.__name__
            tag_value = tagger.tag_value(container)
            LOGGER.info(
                f"Calculated tag, tagger_name: {tagger_name} tag_value: {tag_value}"
            )
            tags.append(f"{registry}/{owner}/{short_image_name}:{platform}-{tag_prefix}-{tag_value}")
    tags_dir.mkdir(parents=True, exist_ok=True)
    (tags_dir / filename).write_text("\n".join(tags))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--short-image-name",
        required=True,
        help="Short image name to write tags for",
    )
    arg_parser.add_argument(
        "--tag-prefix",
        required=True,
        help="Prefix used for all kinds of extra versioning",
    )
    arg_parser.add_argument(
        "--tags-dir",
        required=True,
        type=Path,
        help="Directory to save tags file",
    )
    arg_parser.add_argument(
        "--owner",
        required=True,
        help="Owner of the image",
    )
    arg_parser.add_argument(
        "--registry",
        required=True,
        help="registry for image to apply tags for",
    )
    args = arg_parser.parse_args()

    write_tags_file(args.short_image_name, args.tag_prefix, args.owner, args.tags_dir, args.registry)
