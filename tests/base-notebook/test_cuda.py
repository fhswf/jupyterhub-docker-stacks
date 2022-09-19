import logging

import pytest  # type: ignore

from tests.conftest import TrackedContainer

LOGGER = logging.getLogger(__name__)

@pytest.mark.info
def test_cuda(
    container: TrackedContainer
) -> None:
    LOGGER.info(f"Checking output of nvidia-smi in container: {container.image_name} ...")
    logs = container.run_and_wait(
        timeout=120,  # usermod is slow so give it some time
        tty=True,
        user="root",
        environment=["NB_UID=1010"],
        command=["start.sh", "bash", "-c", "nvidia-smi"],
    )
    assert "CUDA Version" in logs

