"""Synthetic wafer image generator with configurable defect patterns.

Generates realistic wafer images with labeled defects for training
when real data is scarce. Supports all 10 defect classes.
"""

import random
import numpy as np
import cv2
from pathlib import Path
from typing import Optional


DEFECT_COLORS = {
    "scratch": (200, 200, 200),
    "particle": (0, 0, 0),
    "edge_chip": (180, 160, 140),
    "void": (50, 50, 50),
    "pattern_shift": (100, 100, 150),
    "bridge": (220, 200, 180),
    "missing_bond": (80, 80, 80),
    "crack": (150, 130, 110),
    "contamination": (60, 80, 60),
    "delamination": (170, 170, 190),
}


class WaferGenerator:
    """Generate synthetic wafer images with labeled defects.
    
    Creates realistic 640x640 wafer images with:
    - Circular wafer shape with flat edge
    - Grid pattern (die layout)
    - Random defects with bounding box labels
    
    Example:
        >>> gen = WaferGenerator(img_size=640)
        >>> img, labels = gen.generate(n_defects=3)
    """

    def __init__(self, img_size: int = 640, wafer_radius: int = 280):
        self.img_size = img_size
        self.wafer_radius = wafer_radius
        self.center = (img_size // 2, img_size // 2)

    def _draw_wafer_base(self, img: np.ndarray) -> np.ndarray:
        """Draw circular wafer with grid pattern."""
        cx, cy = self.center

        # Wafer background (dark gray)
        cv2.circle(img, self.center, self.wafer_radius, (40, 40, 40), -1)

        # Grid lines (die boundaries)
        grid_spacing = 28
        for x in range(cx - self.wafer_radius, cx + self.wafer_radius, grid_spacing):
            for y in range(cy - self.wafer_radius, cy + self.wafer_radius, grid_spacing):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < self.wafer_radius - 5:
                    cv2.rectangle(img, (x, y), (x + grid_spacing - 2, y + grid_spacing - 2), (55, 55, 55), 1)

        # Flat edge (bottom)
        cv2.rectangle(img, (cx - 40, cy + self.wafer_radius - 5), (cx + 40, cy + self.wafer_radius + 5), (40, 40, 40), -1)

        return img

    def _draw_defect(self, img: np.ndarray, defect_type: str, bbox: tuple) -> np.ndarray:
        """Draw a single defect on the wafer."""
        x, y, w, h = bbox
        color = DEFECT_COLORS.get(defect_type, (200, 200, 200))

        if defect_type == "scratch":
            angle = random.uniform(0, np.pi)
            length = max(w, h)
            x1 = int(x + length * np.cos(angle))
            y1 = int(y + length * np.sin(angle))
            cv2.line(img, (x, y), (x1, y1), color, random.randint(1, 3))

        elif defect_type == "particle":
            cv2.circle(img, (x + w // 2, y + h // 2), min(w, h) // 2, color, -1)

        elif defect_type == "crack":
            points = [(x, y)]
            cx, cy = x, y
            for _ in range(random.randint(3, 8)):
                cx += random.randint(-w // 3, w // 3)
                cy += random.randint(-h // 3, h // 3)
                points.append((cx, cy))
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(img, [pts], False, color, random.randint(1, 3))

        elif defect_type == "void":
            cv2.ellipse(img, (x + w // 2, y + h // 2), (w // 2, h // 2), 0, 0, 360, color, -1)

        elif defect_type == "bridge":
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

        return img

    def generate(
        self,
        n_defects: int = 2,
        defect_types: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic wafer image with defects.
        
        Args:
            n_defects: Number of defects to place.
            defect_types: Specific defect types (random if None).
            
        Returns:
            Tuple of (image, labels) where labels is (N, 5) array:
            [class_id, x_center, y_center, width, height] (normalized).
        """
        if defect_types is None:
            defect_types = list(DEFECT_COLORS.keys())

        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = self._draw_wafer_base(img)

        labels = []
        cx, cy = self.center

        for _ in range(n_defects):
            dtype = random.choice(defect_types)

            # Random position within wafer
            angle = random.uniform(0, 2 * np.pi)
            dist = random.uniform(20, self.wafer_radius - 30)
            dx = int(cx + dist * np.cos(angle))
            dy = int(cy + dist * np.sin(angle))

            # Random size
            w = random.randint(5, 40)
            h = random.randint(5, 40)

            # Draw defect
            self._draw_defect(img, dtype, (dx, dy, w, h))

            # YOLO format label [class_id, x_center, y_center, width, height]
            class_id = list(DEFECT_COLORS.keys()).index(dtype)
            labels.append([
                class_id,
                (dx + w / 2) / self.img_size,
                (dy + h / 2) / self.img_size,
                w / self.img_size,
                h / self.img_size,
            ])

        return img, np.array(labels, dtype=np.float32)

    def generate_dataset(
        self,
        output_dir: str,
        n_images: int = 100,
        min_defects: int = 1,
        max_defects: int = 5,
    ) -> None:
        """Generate a full synthetic dataset.
        
        Args:
            output_dir: Output directory for images and labels.
            n_images: Number of images to generate.
            min_defects: Minimum defects per image.
            max_defects: Maximum defects per image.
        """
        out = Path(output_dir)
        img_dir = out / "images"
        lbl_dir = out / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_images):
            n_def = random.randint(min_defects, max_defects)
            img, labels = self.generate(n_defects=n_def)

            cv2.imwrite(str(img_dir / f"wafer_{i:05d}.png"), img)

            with open(lbl_dir / f"wafer_{i:05d}.txt", "w") as f:
                for lbl in labels:
                    f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        print(f"Generated {n_images} images in {output_dir}")
