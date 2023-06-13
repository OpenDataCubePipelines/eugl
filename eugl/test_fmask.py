import zipfile

import pytest

from .fmask import _extract_file_from_zip


@pytest.fixture
def create_zip_file(tmp_path):
    zip_file = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_file, "w") as archive:
        archive.writestr("dir/file1.xml", "content1")
        archive.writestr("dir/subdir/file2.xml", "content2")
        archive.writestr("dir/subdir/file3.xml", "content3")
        archive.writestr("file4.xml", "content4")
    return zip_file


def test_extract_single_file(create_zip_file, tmp_path):
    destination = tmp_path / "output.xml"
    _extract_file_from_zip(
        zip_file=create_zip_file,
        file_pattern="dir/file1.xml",
        destination_path=destination,
    )
    assert destination.read_text() == "content1"


def test_extract_with_pattern(create_zip_file, tmp_path):
    destination = tmp_path / "output.xml"
    _extract_file_from_zip(
        zip_file=create_zip_file, file_pattern="dir/*.xml", destination_path=destination
    )
    assert destination.read_text() == "content1"


def test_extract_without_directory(create_zip_file, tmp_path):
    destination = tmp_path / "output.xml"
    _extract_file_from_zip(
        zip_file=create_zip_file, file_pattern="*.xml", destination_path=destination
    )
    assert destination.read_text() == "content4"


def test_no_matches(create_zip_file, tmp_path):
    destination = tmp_path / "output.xml"
    with pytest.raises(FileNotFoundError):
        _extract_file_from_zip(
            zip_file=create_zip_file,
            file_pattern="dir/nonexistent.xml",
            destination_path=destination,
        )


def test_multiple_matches(create_zip_file, tmp_path):
    destination = tmp_path / "output.xml"
    with pytest.raises(ValueError):
        _extract_file_from_zip(
            zip_file=create_zip_file,
            file_pattern="dir/subdir/*.xml",
            destination_path=destination,
        )
