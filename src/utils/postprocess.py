"""Post-processing utilities for YOLO detection output.

NMS, confidence filtering, and defect classification.
"""

import numpy as np
import cv2


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Pure NumPy NMS implementation.
    
    Args:
        boxes: (N, 4) array of [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        iou_threshold: IoU threshold for suppression.
        
    Returns:
        List of kept indices.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def filter_by_area(
    boxes: np.ndarray,
    min_area: int = 25,
    max_area: int = 100000,
) -> np.ndarray:
    """Filter detections by bounding box area.
    
    Removes noise (too small) and artifacts (too large).
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mask = (areas >= min_area) & (areas <= max_area)
    return mask


def cluster_defects(
    boxes: np.ndarray,
    distance_threshold: float = 50.0,
) -> list[list[int]]:
    """Group nearby defects into clusters for batch inspection.
    
    Args:
        boxes: (N, 4) bounding boxes.
        distance_threshold: Max pixel distance for clustering.
        
    Returns:
        List of clusters, each containing box indices.
    """
    if len(boxes) == 0:
        return []

    centers = np.column_stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2,
    ])

    visited = set()
    clusters = []

    for i in range(len(boxes)):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)

        for j in range(i + 1, len(boxes)):
            if j in visited:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < distance_threshold:
                cluster.append(j)
                visited.add(j)

        clusters.append(cluster)

    return clusters
