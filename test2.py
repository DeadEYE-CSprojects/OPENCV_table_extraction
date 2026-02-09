import cv2
import numpy as np

def check_page_complexity(image_path):
    """
    Determines if a page is 'Complex' (Tables, Forms, Grids) using OpenCV.
    Cost: $0.00
    Speed: ~0.05s
    Returns: is_complex (bool), 0, 0 (dummy tokens)
    """
    print(f"      -> [OpenCV] Checking layout complexity...")
    
    # 1. Load Image
    img = cv2.imread(image_path, 0) # Load as Grayscale
    if img is None: return False, 0, 0
    
    # 2. Thresholding (Convert to Black & White)
    # Binary Inv: Lines become White, Background becomes Black
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 3. Define Kernels for Line Detection
    # A line must be at least 1/40th of the image width to count
    scale = 40 
    h_kernel_len = np.array(img).shape[1] // scale
    v_kernel_len = np.array(img).shape[0] // scale

    # Horizontal Kernel (Looks for -----)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    # Vertical Kernel (Looks for | )
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))

    # 4. Extract Lines using Morphology
    # Detect Horizontal Lines
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    # Detect Vertical Lines
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # 5. Calculate "Grid Density"
    # Combine lines to find the table structure
    grid = cv2.addWeighted(h_lines, 1, v_lines, 1, 0)
    
    # Find intersections (joints) where horizontal meets vertical
    joints = cv2.bitwise_and(h_lines, v_lines)
    
    # Count the intersection points (Table cells/corners)
    # Non-Zero pixels in 'joints' means an intersection existed
    intersections = cv2.countNonZero(joints)

    # 6. Count Total Lines
    # Find contours of the grid structure
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_lines = len(contours)

    # --- DECISION LOGIC ---
    # Case A: High Intersection Count = DEFINITELY A TABLE
    if intersections > 20: 
        print(f"      -> [Result] Complex (Found Table/Grid with {intersections} joints).")
        return True, 0, 0
        
    # Case B: Many Lines = FORM / SIGNATURE LINES
    # A simple paragraph might have 0-2 dividers. A form has 10+.
    if num_lines > 10:
        print(f"      -> [Result] Complex (Found {num_lines} lines/separators).")
        return True, 0, 0
        
    # Case C: Simple Text
    print(f"      -> [Result] Simple Text (Intersections: {intersections}, Lines: {num_lines}).")
    return False, 0, 0
