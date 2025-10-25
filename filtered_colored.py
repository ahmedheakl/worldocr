import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


def is_colorized_overlay_page(image_path: str, 
                              debug: bool = False) -> Tuple[bool, dict]:
    img = cv2.imread(str(image_path))
    if img is None:
        return False, {}
    
    height, width = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, s, v = cv2.split(hsv)
    s_norm = s / 255.0
    v_norm = v / 255.0
    colored_mask = np.logical_and(s_norm > 0.4, v_norm > 0.25)
    colored_coverage = np.sum(colored_mask) / colored_mask.size
    non_white_mask = v < 240  # Pixels that aren't bright white
    if np.sum(non_white_mask) > 0:
        non_white_saturation = np.mean(s_norm[non_white_mask])
    else:
        non_white_saturation = 0
    row_colors = []
    for i in range(0, height, max(1, height // 50)):  # Sample rows
        row = s[i, :]
        if np.mean(row) > 100:  # High saturation row
            row_colors.append(np.mean(row))
    
    colored_row_ratio = len(row_colors) / max(1, height // max(1, height // 50))
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    if np.sum(colored_mask) > 0:
        colored_pixels = img[colored_mask]
        color_std = np.std(colored_pixels, axis=0).mean()
    else:
        color_std = 0
    colored_mask_uint8 = (colored_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(colored_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    large_colored_regions = 0
    total_colored_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (width * height * 0.01):  # Regions larger than 1% of image
            large_colored_regions += 1
            total_colored_area += area
    
    large_region_coverage = total_colored_area / (width * height)
    metrics = {
        'colored_coverage': colored_coverage,
        'non_white_saturation': non_white_saturation,
        'colored_row_ratio': colored_row_ratio,
        'edge_density': edge_density,
        'color_std': color_std,
        'large_region_coverage': large_region_coverage,
        'num_large_regions': large_colored_regions
    }
    condition1_high_coverage = (
        (colored_coverage > 0.25 or large_region_coverage > 0.20) and
        non_white_saturation > 0.4 and
        edge_density < 0.03 and
        (large_colored_regions >= 2 or large_region_coverage > 0.3)
    )
    condition2_minimal_content = (
        edge_density < 0.005 and
        colored_coverage > 0.05 and
        non_white_saturation > 0.6 and
        large_colored_regions >= 1
    )
    is_overlay = condition1_high_coverage or condition2_minimal_content
    
    if debug:
        print(f"\nMetrics for {Path(image_path).name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"  Decision: {'FILTER' if is_overlay else 'KEEP'}")
    
    return is_overlay, metrics


def is_colorized_overlay_page_simple(image_path: str) -> bool:
    result, _ = is_colorized_overlay_page(image_path, debug=False)
    return result



def filter_documents(image_paths: List[str], debug: bool = False) -> Tuple[List[str], List[str]]:
    normal_pages = []
    filtered_pages = []
    
    for img_path in image_paths:
        is_overlay = is_colorized_overlay_page_simple(img_path)
        
        if is_overlay:
            filtered_pages.append(img_path)
            status = "❌ Filtered"
        else:
            normal_pages.append(img_path)
            status = "✓ Keeping"
        
        if debug:
            _, metrics = is_colorized_overlay_page(img_path, debug=False)
            print(f"{status}: {Path(img_path).name} | "
                  f"Coverage: {metrics['colored_coverage']:.2f}, "
                  f"Sat: {metrics['non_white_saturation']:.2f}, "
                  f"Edges: {metrics['edge_density']:.4f}")
        else:
            print(f"{status}: {Path(img_path).name}")
    
    return normal_pages, filtered_pages


if __name__ == "__main__":
    from glob import glob
    import shutil
    
    
    document_pages = glob("data/omnidocbench_output_large/images/*.jpg")
    batch_size = 64
    for i in range(0, len(document_pages), batch_size):    
        print(f"Processing {i+batch_size} images...\n")
        batch_pages = document_pages[i:i+batch_size]
        normal, filtered = filter_documents(batch_pages, debug=True)
        outroot = "data/omnidocbench_output_large/images/colorized"
        Path(outroot).mkdir(parents=True, exist_ok=True)
        for f in filtered:
            dest_path = Path(outroot) / Path(f).name
            shutil.copy(f, dest_path)
    