import cv2
import numpy as np
import csv
import os
from typing import Dict, List, Optional, Tuple, Any

# --- CONFIGURATION & CONSTANTS ---
DEAD_ZONE_PERCENT: float = 0.3  
OUTPUT_DIR: str = 'detected'
PROFILE_DIR: str = 'profiles'
DIAG_DIR: str = 'diagnostics'
CSV_FILENAME: str = 'vessel_analysis.csv'

FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.5
FONT_THICKNESS: int = 1
COLORS: Dict[str, Tuple[int, int, int]] = {
    'axis': (180, 180, 180),
    'height': (0, 255, 0),
    'belly': (0, 255, 255),
    'rim': (255, 0, 255),
    'wall': (0, 0, 255),
    'profile_highlight': (0, 255, 0),  # Bright Green
    'dead_zone': (0, 0, 255)           # Red
}

def preprocess_image(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    img = cv2.imread(image_path)
    if img is None: return None, None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return img, gray, thresh

def find_central_axis(gray: np.ndarray, x_c: float) -> int:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=gray.shape[0]*0.3, maxLineGap=20)
    axis_x = int(x_c)
    if lines is not None:
        vertical_lines = [(l[0][0] + l[0][2]) / 2 for l in lines if abs(l[0][0] - l[0][2]) < 5]
        if vertical_lines:
            axis_x = int(min(vertical_lines, key=lambda val: abs(val - x_c)))
    return axis_x

def extract_rightmost_outer_profile(thresh: np.ndarray, axis_x: int, opening_width: int, filename: str) -> np.ndarray:
    h, w = thresh.shape
    # 1. Isolate Right Side
    right_mask = np.zeros_like(thresh)
    right_mask[:, axis_x:] = 255
    right_side = cv2.bitwise_and(thresh, right_mask)
    
    # 2. APPLY DEAD ZONE
    rim_radius = opening_width / 2
    dead_zone_limit = axis_x + int(rim_radius * DEAD_ZONE_PERCENT)
    
    dead_zone_visual = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(dead_zone_visual, (axis_x, 0), (dead_zone_limit, h), COLORS['dead_zone'], -1)
    
    right_side_with_dz = right_side.copy()
    right_side_with_dz[:, axis_x:dead_zone_limit] = 0
    
    # 3. Scrub Vertical Lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_structures = cv2.morphologyEx(right_side_with_dz, cv2.MORPH_OPEN, kernel_v)
    right_cleaned = cv2.subtract(right_side_with_dz, vertical_structures)
    
    # 4. Distal Scan & XY Coordinate Capture
    profile_mask = np.zeros_like(right_cleaned)
    coords = []
    for y in range(h):
        on_pixels = np.where(right_cleaned[y, :] == 255)[0]
        if len(on_pixels) > 0:
            x_val = on_pixels[-1]
            profile_mask[y, x_val] = 255
            coords.append((x_val, y))

    # 5. Save Files (Mask & XY Coords)
    if not os.path.exists(PROFILE_DIR): os.makedirs(PROFILE_DIR)
    cv2.imwrite(os.path.join(PROFILE_DIR, f"profile_{filename}"), profile_mask)
    
    coord_file = os.path.join(PROFILE_DIR, f"coords_{os.path.splitext(filename)[0]}.txt")
    with open(coord_file, 'w') as f:
        f.write("x,y\n")
        for x, y in coords:
            f.write(f"{x},{y}\n")
    
    # 6. Create Overlaid Diagnostic Window
    diag_base = cv2.cvtColor(right_side, cv2.COLOR_GRAY2BGR)
    diag_base = cv2.addWeighted(diag_base, 0.3, np.zeros_like(diag_base), 0.7, 0)
    diag_view = cv2.addWeighted(diag_base, 1.0, dead_zone_visual, 0.4, 0)
    
    profile_points = cv2.dilate(profile_mask, np.ones((3,3), np.uint8))
    diag_view[profile_points == 255] = COLORS['profile_highlight']
    
    cv2.putText(diag_view, f"DZ {int(DEAD_ZONE_PERCENT*100)}% | Saved Coords", (axis_x + 5, 25), 
                FONT, 0.5, (255, 255, 255), 1)
    
    # Save Diagnostic Image
    if not os.path.exists(DIAG_DIR): os.makedirs(DIAG_DIR)
    cv2.imwrite(os.path.join(DIAG_DIR, f"diag_{filename}"), diag_view)
    
    return diag_view

def get_row_thickness_data(mask: np.ndarray, y_coord: int) -> Tuple[Optional[int], Optional[int], int]:
    if y_coord >= mask.shape[0]: return None, None, 0
    row = mask[y_coord, :]
    on_pixels = np.where(row == 255)[0]
    if len(on_pixels) >= 2:
        return int(on_pixels[0]), int(on_pixels[-1]), int(on_pixels[-1] - on_pixels[0])
    return None, None, 0

def annotate_render(draw_img: np.ndarray, m: Dict[str, Any]) -> None:
    cv2.line(draw_img, (m['axis_x'], 0), (m['axis_x'], draw_img.shape[0]), COLORS['axis'], 1)
    h_x = int(m['max_x'] + 30)
    cv2.line(draw_img, (h_x, m['min_y']), (h_x, m['max_y']), COLORS['height'], 2)
    cv2.putText(draw_img, f"H: {m['height']}px", (h_x + 5, int(m['min_y'] + m['height']/2)), FONT, FONT_SCALE, COLORS['height'], FONT_THICKNESS)
    cv2.line(draw_img, (m['min_x'], m['belly_y']), (m['max_x'], m['belly_y']), COLORS['belly'], 2)
    cv2.putText(draw_img, f"Belly: {m['belly_width']}px", (m['min_x'], m['belly_y'] - 10), FONT, FONT_SCALE, COLORS['belly'], FONT_THICKNESS)
    cv2.line(draw_img, (int(m['axis_x'] - m['opening_width']/2), m['min_y']), (int(m['axis_x'] + m['opening_width']/2), m['min_y']), COLORS['rim'], 2)
    cv2.putText(draw_img, f"Rim: {m['opening_width']}px", (int(m['axis_x'] - m['opening_width']/2), m['min_y'] - 10), FONT, FONT_SCALE, COLORS['rim'], FONT_THICKNESS)

def analyze_vessel(image_path: str) -> Optional[Dict[str, Any]]:
    img_name = os.path.basename(image_path)
    img, gray, thresh = preprocess_image(image_path)
    if img is None or thresh is None: return None

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    largest_contour = max(contours, key=cv2.contourArea)
    (x_c, y_c), _ = cv2.minEnclosingCircle(largest_contour)
    axis_x = find_central_axis(gray, x_c)
    
    points = cv2.findNonZero(thresh)
    hull = cv2.convexHull(points); hull_pts = hull.reshape(-1, 2)
    min_y, max_y = int(np.min(hull_pts[:, 1])), int(np.max(hull_pts[:, 1]))
    top_pts = hull_pts[hull_pts[:, 1] <= min_y + ((max_y - min_y) * 0.05)]
    opening_width = (max([abs(p[0] - axis_x) for p in top_pts]) * 2) if len(top_pts) > 0 else 0

    # 1. Profile Extraction & Diagnostic Save
    diag_view = extract_rightmost_outer_profile(thresh, axis_x, opening_width, img_name)
    cv2.imshow("Diagnostic (Saved to /diagnostics)", diag_view)

    # 2. Measurements
    min_x, max_x = int(np.min(hull_pts[:, 0])), int(np.max(hull_pts[:, 0]))
    belly_y = int(hull_pts[np.argmax(hull_pts[:, 0]), 1])
    left_mask = np.zeros_like(thresh); left_mask[:, :axis_x] = 255
    left_only = cv2.bitwise_and(thresh, left_mask)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    wall_only = cv2.subtract(left_only, cv2.morphologyEx(left_only, cv2.MORPH_OPEN, kernel_v))
    b_start, b_end, b_thick = get_row_thickness_data(wall_only, belly_y)

    m_data = {
        'axis_x': axis_x, 'min_y': min_y, 'max_y': max_y, 'min_x': min_x, 'max_x': max_x,
        'belly_y': belly_y, 'height': max_y - min_y, 'belly_width': max_x - min_x,
        'opening_width': int(opening_width), 'b_start': b_start, 'b_end': b_end, 'belly_wall': b_thick
    }

    draw_img = img.copy()
    annotate_render(draw_img, m_data)
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"vessel_{img_name}"), draw_img)
    cv2.imshow("Final Analysis", draw_img)
    
    cv2.waitKey(0)
    return {"filename": img_name, "height_px": m_data['height'], "belly_width_px": m_data['belly_width'], 
            "opening_width_px": m_data['opening_width'], "belly_wall_px": m_data['belly_wall']}

def batch_process(folder_path: str) -> None:
    all_data = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for f in files:
        data = analyze_vessel(os.path.join(folder_path, f))
        if data: all_data.append(data)
    if all_data:
        with open(CSV_FILENAME, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
            writer.writeheader(); writer.writerows(all_data)

if __name__ == "__main__":
    batch_process("input")