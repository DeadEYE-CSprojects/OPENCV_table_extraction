# =============================================================================
# ADD/UPDATE THESE FUNCTIONS IN SECTION 2
# =============================================================================

def detect_and_crop_grid(image, target_size=(2480, 3508)):
    """
    New Function: Detects if a large grid/table exists and crops to it.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Binary Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Detect Horizontal Lines
    hor_kernel_len = np.array(image).shape[1] // 40
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_kernel_len, 1))
    img_hor = cv2.erode(thresh, hor_kernel, iterations=1)
    img_hor = cv2.dilate(img_hor, hor_kernel, iterations=1)

    # 2. Detect Vertical Lines
    ver_kernel_len = np.array(image).shape[0] // 40
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_kernel_len))
    img_ver = cv2.erode(thresh, ver_kernel, iterations=1)
    img_ver = cv2.dilate(img_ver, ver_kernel, iterations=1)

    # 3. Combine to find the Grid Structure
    grid_mask = cv2.addWeighted(img_hor, 0.5, img_ver, 0.5, 0.0)
    _, grid_mask = cv2.threshold(grid_mask, 0, 255, cv2.THRESH_BINARY)
    
    # 4. Find Contours
    cnts, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        largest_cnt = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_cnt)
        
        # Validation: Grid must be > 10% of image
        if (cw * ch) > (0.1 * w * h):
            pad = 15
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)
            
            cropped = image[y1:y2, x1:x2]
            return True, cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return False, None

def find_anchors_and_crop(image, target_size=(2480, 3508)):
    """
    Updated Function: Crops based on Top-Left Object and Bottom-Right Object.
    (Replaces the old QR-specific logic)
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Connect elements
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        return cv2.resize(image, target_size)

    # 1. Find Top-Left Anchor (Smallest x + y)
    # Only look in top-left quadrant
    tl_candidates = [c for c in cnts if cv2.boundingRect(c)[0] < w//2 and cv2.boundingRect(c)[1] < h//2]
    if tl_candidates:
        tl_cnt = min(tl_candidates, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1])
        x1, y1, _, _ = cv2.boundingRect(tl_cnt)
    else:
        x1, y1 = 0, 0

    # 2. Find Bottom-Right Anchor (Largest x + y + w + h)
    # Only look in bottom-right quadrant
    br_candidates = [c for c in cnts if cv2.boundingRect(c)[0] > w//3 and cv2.boundingRect(c)[1] > h//3]
    if br_candidates:
        br_cnt = max(br_candidates, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] + cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3])
        bx, by, bw, bh = cv2.boundingRect(br_cnt)
        x2, y2 = bx + bw, by + bh
    else:
        x2, y2 = w, h

    # 3. Crop
    pad = 20
    final_x1 = max(0, x1 - pad)
    final_y1 = max(0, y1 - pad)
    final_x2 = min(w, x2 + pad)
    final_y2 = min(h, y2 + pad)
    
    if final_x2 > final_x1 and final_y2 > final_y1:
        cropped = image[final_y1:final_y2, final_x1:final_x2]
        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return cv2.resize(image, target_size)


# =============================================================================
# UPDATE THIS FUNCTION IN SECTION 4
# =============================================================================

def run_stage_1_preprocessing():
    """Converts TIFF -> Deskewed -> Grid/Anchor Crop -> PNG."""
    print("\n--- STAGE 1: Pre-processing (TIFF to PNG) ---")
    os.makedirs(INTERMEDIATE_PNG_FOLDER, exist_ok=True)
    
    files = glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.tif*')) + \
            glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.TIF*'))
    
    if not files:
        print("No TIFF files found.")
        return

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        base_name = os.path.splitext(fname)[0]
        print(f"Processing {i+1}/{len(files)}: {fname}")
        
        img = cv2.imread(fpath)
        if img is None: continue
            
        # 1. Deskew
        deskewed = deskew_single_image(img)
        
        # 2. Check for Grid
        is_grid, grid_cropped = detect_and_crop_grid(deskewed)
        
        final_img = None
        prefix = ""
        
        if is_grid:
            print("   -> Type: GRID Detected")
            prefix = "Grid_"
            final_img = grid_cropped
        else:
            print("   -> Type: NO GRID (Using Anchor crop)")
            prefix = "NGrid_"
            final_img = find_anchors_and_crop(deskewed)
        
        # 3. Save
        save_name = f"{prefix}{base_name}.png"
        cv2.imwrite(os.path.join(INTERMEDIATE_PNG_FOLDER, save_name), final_img)


