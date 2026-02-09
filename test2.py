import cv2
import numpy as np

def check_page_complexity(image_path):
    """
    SENSITIVE MODE: Detects 2x2 tables, 2-column layouts, and small grids.
    """
    # 1. Load & Preprocess
    img = cv2.imread(image_path, 0)
    if img is None: return False, 0, 0
    
    # Invert (Text/Lines = White, BG = Black)
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # --- STRATEGY 1: GRID DETECTION (For Tables with Lines) ---
    scale = 30 # Slightly more sensitive to short lines
    h, w = img.shape[:2]
    
    # Define Kernels
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // scale, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // scale))

    # Detect Lines
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Find Intersections (Joints)
    joints = cv2.bitwise_and(h_lines, v_lines)
    intersections = cv2.countNonZero(joints)

    # Find "Cells" (Closed boxes formed by lines)
    grid = cv2.add(h_lines, v_lines)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Count valid cells (Filter out tiny noise dots)
    cells = 0
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        # A cell must be somewhat box-shaped (bigger than 20x20 pixels)
        if w_box > 20 and h_box > 20: 
            cells += 1

    # --- STRATEGY 2: WHITESPACE COLUMN DETECTION (For Text Columns) ---
    # If there are NO lines, we check if text is split into 2 blocks
    has_columns = False
    if cells < 2:
        # Smear text horizontally to connect words into "blobs"
        kernel_fat = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(thresh, kernel_fat, iterations=3)
        
        # Calculate Vertical Projection (Sum of white pixels down each column)
        col_sum = np.sum(dilated, axis=0)
        
        # Look for a "gap" in the middle 50% of the page
        middle_start = w // 3
        middle_end = 2 * w // 3
        middle_section = col_sum[middle_start:middle_end]
        
        # If we find a slice of 0s (black) in the middle, it's a column gap
        if np.min(middle_section) == 0:
            has_columns = True

    # --- FINAL DECISION ---
    
    # 1. Is it a 2x2 Table? (Needs ~4 intersections or ~4 cells)
    if intersections > 4 or cells > 3:
        print(f"      -> [Complex] Table Detected ({cells} cells, {intersections} joints).")
        return True, 0, 0
        
    # 2. Is it a 2-Column Layout (Invisible Lines)?
    if has_columns:
        print(f"      -> [Complex] Multi-Column Text Detected (Whitespace gap found).")
        return True, 0, 0

    # 3. Fallback
    print(f"      -> [Simple] Standard Text ({cells} cells).")
    return False, 0, 0
