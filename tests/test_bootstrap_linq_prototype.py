from pathlib import Path
import tempfile
import zipfile

import pytest

from scripts.bootstrap_linq_prototype import validate_protocol_wheel


def write_wheel(path, metadata):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('tpu_pycloud_format-0.0.1.dist-info/METADATA', metadata)
    return path


def test_accepts_lin_q_supplied_protocol_wheel_metadata():
    with tempfile.TemporaryDirectory() as directory:
        wheel = write_wheel(Path(directory) / 'protocol.whl',
                            'Metadata-Version: 2.1\nName: tpu-pycloud-format\nVersion: 0.0.1\n')
        assert validate_protocol_wheel(wheel) == wheel


@pytest.mark.parametrize('metadata', [
    'Metadata-Version: 2.1\nName: unrelated-package\nVersion: 0.0.1\n',
    'Metadata-Version: 2.1\nName: tpu-pycloud-format\nVersion: 9.9.9\n',
])
def test_rejects_other_protocol_wheel_metadata(metadata):
    with tempfile.TemporaryDirectory() as directory:
        wheel = write_wheel(Path(directory) / 'protocol.whl', metadata)
        with pytest.raises(ValueError, match='tpu_pycloud_format version 0.0.1'):
            validate_protocol_wheel(wheel)


def test_rejects_missing_protocol_wheel_before_bootstrap():
    with pytest.raises(FileNotFoundError, match='LinQ-supplied'):
        validate_protocol_wheel('/does/not/exist.whl')
