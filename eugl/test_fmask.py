import os
import shutil
import tarfile
import zipfile

import pytest

from .fmask import FileArchive


@pytest.fixture(params=["zip", "tar", "dir"])
def create_archive_file(tmp_path, request):
    archive_file = tmp_path / f"test.{request.param}"

    # Initialize the files with content
    (tmp_path / "file1.xml").write_text("content1")
    (tmp_path / "file2.xml").write_text("content2")
    (tmp_path / "file3.xml").write_text("content3")
    (tmp_path / "file4.xml").write_text("content4")

    if request.param == "zip":
        with zipfile.ZipFile(archive_file, "w") as archive:
            archive.write(tmp_path / "file1.xml", arcname="dir/file1.xml")
            archive.write(tmp_path / "file2.xml", arcname="dir/subdir/file2.xml")
            archive.write(tmp_path / "file3.xml", arcname="dir/subdir/file3.xml")
            archive.write(tmp_path / "file4.xml", arcname="file4.xml")
    elif request.param == "tar":
        with tarfile.open(archive_file, "w") as archive:
            archive.add(tmp_path / "file1.xml", arcname="dir/file1.xml")
            archive.add(tmp_path / "file2.xml", arcname="dir/subdir/file2.xml")
            archive.add(tmp_path / "file3.xml", arcname="dir/subdir/file3.xml")
            archive.add(tmp_path / "file4.xml", arcname="file4.xml")
    elif request.param == "dir":
        os.makedirs(archive_file / "dir/subdir", exist_ok=True)
        shutil.move(tmp_path / "file1.xml", archive_file / "dir/file1.xml")
        shutil.move(tmp_path / "file2.xml", archive_file / "dir/subdir/file2.xml")
        shutil.move(tmp_path / "file3.xml", archive_file / "dir/subdir/file3.xml")
        shutil.move(tmp_path / "file4.xml", archive_file / "file4.xml")
    return archive_file


def test_extract_single_file(create_archive_file, tmp_path):
    destination = tmp_path / "output.xml"
    with FileArchive(create_archive_file) as archive:
        archive.extract_file(file_pattern="dir/file1.xml", destination_path=destination)
    assert destination.read_text() == "content1"


def test_extract_with_pattern(create_archive_file, tmp_path):
    destination = tmp_path / "output.xml"
    with FileArchive(create_archive_file) as archive:
        archive.extract_file(file_pattern="dir/*.xml", destination_path=destination)
    assert destination.read_text() == "content1"


def test_extract_without_directory(create_archive_file, tmp_path):
    destination = tmp_path / "output.xml"
    with FileArchive(create_archive_file) as archive:
        archive.extract_file(file_pattern="*.xml", destination_path=destination)
    assert destination.read_text() == "content4"


def test_no_matches(create_archive_file, tmp_path):
    destination = tmp_path / "output.xml"
    with pytest.raises(FileNotFoundError):
        with FileArchive(create_archive_file) as archive:
            archive.extract_file(
                file_pattern="dir/nonexistent.xml", destination_path=destination
            )


def test_multiple_matches(create_archive_file, tmp_path):
    destination = tmp_path / "output.xml"
    with pytest.raises(ValueError):
        with FileArchive(create_archive_file) as archive:
            archive.extract_file(
                file_pattern="dir/subdir/*.xml", destination_path=destination
            )
