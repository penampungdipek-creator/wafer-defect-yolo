"""Advanced augmentation pipeline for wafer defect images.

Implements Mosaic, MixUp, copy-paste defect synthesis, and
domain-specific transforms for semiconductor imagery.
"""

import random
import numpy as np
import cv2
from typing import Optional


class WaferAugmentation:
    """Augmentation pipeline optimized for wafer defect detection.
    
    Combines general object detection augmentations with
    semiconductor-specific transforms:
    - Mosaic (4-image composition)
    - MixUp (image blending)
    - Copy-paste defect synthesis
    - Brightness/contrast for microscope images
    - Gaussian noise (sensor noise simulation)
    """

    def __init__(
        self,
        img_size: int = 640,
        mosaic_prob: float = 1.0,
        mixup_prob: float = 0.15,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 10.0,
        translate: float = 0.1,
        scale: float = 0.5,
        shear: float = 2.0,
    ):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def mosaic(self, images: list[np.ndarray], labels: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """4-image Mosaic augmentation.
        
        Combines 4 images into one, simulating multi-defect scenes.
        """
        s = self.img_size
        cx, cy = random.randint(s // 4, s * 3 // 4), random.randint(s // 4, s * 3 // 4)

        result_img = np.full((s, s, 3), 114, dtype=np.uint8)
        all_labels = []

        positions = [(0, 0), (s, 0), (0, s), (s, s)]
        for i, (img, lbl) in enumerate(zip(images, labels)):
            h, w = img.shape[:2]
            px, py = positions[i]

            # Resize to fit quadrant
            scale = min(s // 2 / h, s // 2 / w)
            img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))

            rh, rw = img_resized.shape[:2]
            x1 = max(0, px - rw)
            y1 = max(0, py - rh)
            x2 = min(s, px)
            y2 = min(s, py)

            result_img[y1:y2, x1:x2] = img_resized[:y2-y1, :x2-x1]

            # Adjust labels
            if len(lbl) > 0:
                lbl_shifted = lbl.copy()
                lbl_shifted[:, 1] = (lbl_shifted[:, 1] * scale + x1) / s
                lbl_shifted[:, 2] = (lbl_shifted[:, 2] * scale + y1) / s
                lbl_shifted[:, 3] = lbl_shifted[:, 3] * scale / s
                lbl_shifted[:, 4] = lbl_shifted[:, 4] * scale / s
                all_labels.append(lbl_shifted)

        combined_labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, 5))
        return result_img, combined_labels

    def mixup(self, img1: np.ndarray, img2: np.ndarray, alpha: float = 1.5) -> np.ndarray:
        """MixUp augmentation — blend two images."""
        beta = np.random.beta(alpha, alpha)
        return cv2.addWeighted(img1, beta, img2, 1 - beta, 0)

    def hsv_augment(self, img: np.ndarray) -> np.ndarray:
        """HSV color space augmentation (microscope lighting simulation)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] += random.uniform(-self.hsv_h, self.hsv_h) * 180
        hsv[:, :, 1] *= random.uniform(1 - self.hsv_s, 1 + self.hsv_s)
        hsv[:, :, 2] *= random.uniform(1 - self.hsv_v, 1 + self.hsv_v)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def add_gaussian_noise(self, img: np.ndarray, sigma: float = 10.0) -> np.ndarray:
        """Simulate sensor noise from microscope cameras."""
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def copy_paste_defect(
        self,
        background: np.ndarray,
        defect_patch: np.ndarray,
        position: tuple[int, int],
    ) -> np.ndarray:
        """Copy-paste defect synthesis.
        
        Pastes a defect patch onto a clean wafer region.
        """
        h, w = defect_patch.shape[:2]
        x, y = position
        x2, y2 = min(x + w, background.shape[1]), min(y + h, background.shape[0])

        mask = defect_patch[:y2-y, :x2-x].mean(axis=2) > 20  # Non-black pixels
        result = background.copy()
        for c in range(3):
            result[y:y2, x:x2, c] = np.where(
                mask,
                defect_patch[:y2-y, :x2-x, c],
                background[y:y2, x:x2, c],
            )
        return result

    def __call__(self, img: np.ndarray, labels: Optional[np.ndarray] = None) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random augmentations."""
        img = self.hsv_augment(img)

        if random.random() < 0.3:
            img = self.add_gaussian_noise(img, sigma=random.uniform(5, 15))

        return img, labels
