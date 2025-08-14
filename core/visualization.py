from typing import List, Tuple

import cv2
import numpy as np
from torch import Tensor


def draw_density_based_result(
    image: np.ndarray,
    density_map: Tensor,
    count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Visualize results for density-based models.

    Args:
        image: Original image (numpy array, HWC, normalized)
        density_map: Predicted density map (2D numpy array)
        count: Predicted count

    Returns:
        Blended visualization image
    """
    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize and colormap density
    density_map = density_map[0, 0].detach().cpu().numpy()
    density_normalized = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    density_colored = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)
    density_map_final = cv2.resize(density_colored, (image.shape[1], image.shape[0]))

    # overlay original image with density map
    mask = density_map_final > 0
    overlay_weight = 0.6
    overlay = image.copy()
    overlay[mask] = cv2.addWeighted(image, 1 - overlay_weight, density_map_final, overlay_weight, 0)[mask]

    # Add count text
    cv2.putText(overlay, f"Count:{count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return density_colored, overlay

def draw_point_based_result(
    image: np.ndarray,
    points: List[tuple],
    count: int
) -> np.ndarray:
    """
    Visualize results for point-based models.

    Args:
        image: Original image (numpy array, HWC, normalized)
        points: List of (x, y) point coordinates
        count: Predicted count

    Returns:
        Annotated visualization image
    """
    # Create a copy to draw on
    annotated = image.copy()

    # Draw points
    for point in points:
        cv2.circle(annotated, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 255), thickness=-1)

    # Add count text
    cv2.putText(annotated, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return annotated