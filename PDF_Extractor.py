def crop_ub04(image):
    """
    Crops a UB04 form. Handles both grid and non-grid versions.
    It first tries to find a table grid. If that fails, it crops
    based on the bounding box of all text content.
    """
    # --- METHOD 1: ATTEMPT GRID-BASED CROPPING ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    grid_mask = cv2.add(detected_horizontal, detected_vertical)
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if a reasonably large grid contour was found
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 50000: # Threshold for grid area
        print("    - Grid detected. Cropping to table boundaries.")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]

    # --- METHOD 2: FALLBACK TO TEXT-BASED CROPPING ---
    else:
        print("    - No grid detected. Cropping based on text content.")
        # Use Pytesseract to get data on all text
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        boxes = len(data['level'])
        # Filter out empty or low-confidence text boxes
        x_coords, y_coords, widths, heights = [], [], [], []
        for i in range(boxes):
            # Only consider boxes with actual text and decent confidence
            if int(data['conf'][i]) > 60 and data['text'][i].strip() != "":
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                x_coords.append(x)
                y_coords.append(y)
                widths.append(w)
                heights.append(h)
        
        if not x_coords:
            print("    - Error: No text found for text-based cropping.")
            return None
            
        # Find the overall bounding box for all text
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max([x + w for x, w in zip(x_coords, widths)])
        max_y = max([y + h for y, h in zip(y_coords, heights)])
        
        # Add a small padding to the crop
        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.shape[1], max_x + padding)
        max_y = min(image.shape[0], max_y + padding)
        
        return image[min_y:max_y, min_x:max_x]

import cv2
import numpy as np

def crop_hbcf(image):
    """
    Crops an HBCF form based on the intersection of lines.
    Top-Right Corner: The uppermost, rightmost intersection.
    Bottom-Left Corner: The lowermost, leftmost intersection.
    """
    # 1. Preprocess the image to detect lines clearly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use adaptive thresholding for better results on varying lighting
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        print("    - Error: No lines were detected in the image.")
        return None

    # 2. Separate lines into horizontal and vertical lists
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) < 5:  # Horizontal line
            horizontal_lines.append(line[0])
        elif abs(x1 - x2) < 5:  # Vertical line
            vertical_lines.append(line[0])

    if not horizontal_lines or not vertical_lines:
        print("    - Error: Could not find both horizontal and vertical lines.")
        return None

    # 3. Find all intersection points between horizontal and vertical lines
    intersections = []
    for h_line in horizontal_lines:
        hx1, hy1, hx2, _ = h_line
        # Use the average y for the horizontal line's position
        h_y = (hy1) 
        
        for v_line in vertical_lines:
            vx1, vy1, _, vy2 = v_line
            # Use the average x for the vertical line's position
            v_x = (vx1) 
            
            # Check if the vertical line's x is within the horizontal line's bounds
            # and the horizontal line's y is within the vertical line's bounds
            if min(hx1, hx2) <= v_x <= max(hx1, hx2) and min(vy1, vy2) <= h_y <= max(vy1, vy2):
                intersections.append((v_x, h_y))

    if not intersections:
        print("    - Error: No line intersections were found.")
        return None
        
    # 4. Find the two specific corners as per the rules
    # "left lower most intersection"
    bottom_left_point = max(intersections, key=lambda p: (p[1], -p[0]))
    
    # "rightside uppermost intersection"
    top_right_point = min(intersections, key=lambda p: (p[1], -p[0]))
    
    x_bl, y_bl = bottom_left_point
    x_tr, y_tr = top_right_point

    # Add a small padding to ensure the lines themselves are included
    padding = 5
    top = max(0, y_tr - padding)
    bottom = min(image.shape[0], y_bl + padding)
    left = max(0, x_bl - padding)
    right = min(image.shape[1], x_tr + padding)
    
    # Final check to ensure coordinates are valid
    if top >= bottom or left >= right:
        print("    - Error: Calculated crop dimensions are invalid.")
        return None
        
    print(f"    - Cropping based on intersections: TL({left},{top}) BR({right},{bottom})")
    
    # 5. Perform the crop
    return image[top:bottom, left:right]





