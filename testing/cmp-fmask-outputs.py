#!/usr/bin/env python3

import argparse
from collections import defaultdict
from textwrap import indent

import math
import numpy as np
import sys
import yaml
from click import secho, echo
from pathlib import Path
from typing import List, Dict, Optional


def compare_yaml_dicts(data1: Dict, data2: Dict, key_prefix="") -> List[str]:
    differences = []

    for key, value in data1.items():
        current_key = f"{key_prefix}.{key}" if key_prefix else key

        if isinstance(value, dict):
            differences.extend(compare_yaml_dicts(value, data2.get(key, {}), current_key))
        else:
            new_value = data2.get(key)

            # Handle NaN values
            if (
                isinstance(value, float)
                and math.isnan(value)
                and isinstance(new_value, float)
                and math.isnan(new_value)
            ):
                continue

            if value != new_value:
                if isinstance(value, (int, float, str, bool)) and isinstance(
                    new_value, type(value)
                ):
                    differences.append(
                        f"{current_key} changed from {value} to {new_value}"
                    )
                else:
                    differences.append(f"{current_key} is different")

    return differences


def compare_yaml_files(
    file1: Path, file2: Path, ignore=("software_versions",)
) -> List[str]:
    with open(file1) as f1, open(file2) as f2:
        data1 = yaml.safe_load(f1)
        data2 = yaml.safe_load(f2)

    for key in ignore:
        data1.pop(key, None)
        data2.pop(key, None)

    return compare_yaml_dicts(data1, data2)


def compare_images_plainly(image1: Path, image2: Path) -> str:
    import rasterio

    with rasterio.open(image1) as img1_ds:
        img1 = img1_ds.read()
    with rasterio.open(image2) as img2_ds:
        img2 = img2_ds.read()

    if img1.shape != img2.shape:
        return f"Image dimensions differ: {img1.shape} vs {img2.shape}"

    total_pixels = np.prod(img1.shape)
    pixels_diff = img1 != img2
    count_diff = np.sum(pixels_diff)
    percentage_diff = (count_diff / total_pixels) * 100
    magnitude_diff = np.sum(np.abs(img1 - img2)) / count_diff if count_diff > 0 else 0

    if count_diff > 0:
        return (
            f"{count_diff} pixels ({percentage_diff:.2f}%) differ "
            f"with avg magnitude of {magnitude_diff:.6f}"
        )
    else:
        return ""


def compare_fmask_images(image1: Path, image2: Path) -> Optional[str]:
    import rasterio

    with rasterio.open(image1) as img1_ds:
        img1 = img1_ds.read()
    with rasterio.open(image2) as img2_ds:
        img2 = img2_ds.read()

    if img1.shape != img2.shape:
        return f"Image dimensions differ: {img1.shape} vs {img2.shape}"

    total_pixels = np.prod(img1.shape)
    pixels_diff = img1 != img2
    count_diff = np.sum(pixels_diff)

    difference_classification = defaultdict(int)
    classification_names = {
        0: "Invalid",
        1: "Clear",
        2: "Cloud",
        3: "Cloud-Shadow",
        4: "Snow",
        5: "Water",
    }

    # If there are differences, gather more information
    if count_diff > 0:
        diff_locations = np.where(pixels_diff)
        avg_magnitude = np.sum(np.abs(img1 - img2)) / count_diff

        # Spatial bins to count differences
        num_bins = 10
        x_bin_size = img1.shape[1] // num_bins
        y_bin_size = img1.shape[2] // num_bins
        spatial_bins = np.zeros((num_bins, num_bins), dtype=int)

        # For each different pixel
        for idx in zip(*diff_locations):
            val1, val2 = img1[idx], img2[idx]
            # Classify the differences
            difference_classification[(val1, val2)] += 1
            # Update the bin
            x_bin = idx[1] // x_bin_size
            y_bin = idx[2] // y_bin_size
            spatial_bins[x_bin, y_bin] += 1

        # Print how each category changed (eg: "50 from cloud to water")
        classification_summary = "\n".join(
            [
                f"{classification_names.get(k[0], 'Unknown')} -> "
                f"{classification_names.get(k[1], 'Unknown')}: {v}"
                for k, v in sorted(difference_classification.items())
            ]
        )

        result = (
            f"{count_diff} pixels ({(count_diff / total_pixels) * 100:.2f}%) "
            f"differ with avg magnitude of {avg_magnitude:.6f}\n"
        )
        result += f"Differences:\n{indent(classification_summary, '  ')}\n"

        # ASCII-art table for spatial bins, for fun. (were all pixels in the edges? etc)
        result += "Spatial distribution of differences:\n"
        max_count_len = len(str(np.max(spatial_bins)))
        cell_width = max(5, max_count_len + 2)
        for i in range(num_bins):
            for j in range(num_bins):
                cell_content = str(spatial_bins[i, j])
                padding = cell_width - len(cell_content)
                left_padding = padding // 2
                right_padding = padding - left_padding
                result += f"|{' ' * left_padding}{cell_content}{' ' * right_padding}"
            result += "|\n"

        return result
    return None


def compare_test_subdirectories(directory1: Path, directory2: Path) -> List[str]:
    differences = []

    yaml1 = directory1 / "fmask.yaml"
    yaml2 = directory2 / "fmask.yaml"
    img1 = directory1 / "fmask.img"
    img2 = directory2 / "fmask.img"

    if yaml2.is_file():
        diff = compare_yaml_files(yaml1, yaml2)
        if diff:
            differences.extend(diff)
    else:
        differences.append("Missing fmask.yaml")

    if img2.is_file():
        diff = compare_fmask_images(img1, img2)
        if diff:
            differences.append(diff)
    else:
        differences.append("Missing fmask.img")

    return differences


def main(root_orig_path: str, root_new_path: str):
    root_orig = Path(root_orig_path)
    root_new = Path(root_new_path)

    failed_comparisons = 0

    for test_dataset in root_orig.iterdir():
        directory1 = test_dataset
        directory2 = root_new / test_dataset.name

        if not (directory1 / "fmask.yaml").is_file():
            # Original data couldn't process it.
            continue

        if not directory2.is_dir():
            secho(f"Missing {test_dataset.name}:", bold=True)
            continue

        secho(f"Differences in {test_dataset.name}:", bold=True)
        differences = compare_test_subdirectories(directory1, directory2)
        prefix = " " * 4
        if differences:
            failed_comparisons += 1
            for diff in differences:
                echo(indent(diff, prefix))
        else:
            echo(indent("None", prefix))

    sys.exit(failed_comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fmask outputs for differences")
    parser.add_argument("original_directory", help="Path to the original directory")
    parser.add_argument("new_directory", help="Path to the new directory")

    args = parser.parse_args()

    main(args.original_directory, args.new_directory)
