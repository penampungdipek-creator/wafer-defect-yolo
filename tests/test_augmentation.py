"""Tests for data augmentation pipeline."""

import numpy as np
import pytest


def test_wafer_generator():
    """Test synthetic wafer generation."""
    from src.data.wafer_generator import WaferGenerator

    gen = WaferGenerator(img_size=640)
    img, labels = gen.generate(n_defects=3)

    assert img.shape == (640, 640, 3)
    assert labels.shape[0] == 3
    assert labels.shape[1] == 5  # class_id, cx, cy, w, h

    # Check normalized coordinates
    for lbl in labels:
        assert 0 <= lbl[1] <= 1  # x_center
        assert 0 <= lbl[2] <= 1  # y_center
        assert 0 <= lbl[3] <= 1  # width
        assert 0 <= lbl[4] <= 1  # height


def test_wafer_generator_no_defects():
    """Test generation with zero defects."""
    from src.data.wafer_generator import WaferGenerator

    gen = WaferGenerator(img_size=640)
    img, labels = gen.generate(n_defects=0)

    assert img.shape == (640, 640, 3)
    assert len(labels) == 0


def test_augmentation_preserves_shape():
    """Test augmentation doesn't change image shape."""
    from src.data.augment import WaferAugmentation

    aug = WaferAugmentation(img_size=640)
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    result, _ = aug(img)
    assert result.shape == img.shape
