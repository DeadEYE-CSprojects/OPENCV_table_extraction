import cv2
import numpy as np
import os
import glob
import pandas as pd
import pytesseract
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

# =============================================================================
# 1. USER CONFIGURATION (EDIT THIS SECTION)
# =============================================================================

# --- A. PATHS ---
INPUT_TIFF_FOLDER = 'raw_tiff_images'        # Folder with your source TIFFs
INTERMEDIATE_PNG_FOLDER = 'processed_pngs'   # Folder to save deskewed/cropped PNGs
FINAL_OUTPUT_FOLDER = 'final_data'           # Folder for cell crops and Excel report

# --- B. TESSERACT PATH (Required for Windows) ---
# If you are on Windows, uncomment and set the path to tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- C. CELL COORDINATES (For the 8 Crops) ---
# Find these coordinates (x1, y1, x2, y2) using the 'processed_pngs' images.
# Methods: 'default', 'digits_only', 'signature_check'
CELL_CONFIG = {
    '1_PatientName': {'coords': (100, 100, 500, 200), 'method': 'default'},
    '2_PolicyNum':   {'coords': (100, 250, 500, 350), 'method': 'digits_only'},
    '3_DateOfServ':  {'coords': (100, 400, 500, 500), 'method': 'default'},
    '4_Diagnosis':   {'coords': (100, 550, 500, 650), 'method': 'default'},
    '5_TotalAmt':    {'coords': (600, 100, 900, 200), 'method': 'digits_only'},
    '6_ProviderID':  {'coords': (600, 250, 900, 350), 'method': 'digits_only'},
    '7_Signature':   {'coords': (600, 400, 900, 500), 'method': 'signature_check'},
    '8_Notes':       {'coords': (600, 550, 900, 650), 'method': 'default'},
}

# =============================================================================
# 2. HELPER FUNCTIONS: IMAGE PROCESSING
# =============================================================================

def deskew_single_image(image):
    """Deskews an image using projection profiles."""
    if image is None: return None
    h, w = image.shape[:2]
    
    # Downscale for speed
    scale = 800 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    scores = []
    angles = np.arange(-5, 5.1, 0.1)
    
    for angle in angles:
        rotated = rotate(thresh, angle, reshape=False, order=0)
        score = np.var(np.sum(rotated, axis=1))
        scores.append(score)
    
    best_angle = angles[np.argmax(scores)]
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def find_anchors_and_crop(image, target_size=(2480, 3508)):
    """Crops image based on Top-Left QR and Bottom-Right Text."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 1. Top-Left Anchor (QR)
    tl_search = thresh[0:h//3, 0:w//2] 
    qr_detector = cv2.QRCodeDetector()
    retval, _, points, _ = qr_detector.detectAndDecodeMulti(tl_search)
    
    x1, y1 = 0, 0
    if retval:
        x1 = int(np.min(points[0][:, 0]))
        y1 = int(np.min(points[0][:, 1]))
    else:
        cnts, _ = cv2.findContours(tl_search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            bx, by, _, _ = cv2.boundingRect(largest)
            x1, y1 = bx, by

    # 2. Bottom-Right Anchor (Text)
    br_search = thresh[2*h//3:h, w//2:w]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(br_search, kernel, iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x2, y2 = w, h
    if cnts:
        best_cnt = max(cnts, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3])
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        x2 = bx + bw + (w // 2)
        y2 = by + bh + (2 * h // 3)

    # 3. Crop & Resize
    pad = 20
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    
    if cx2 > cx1 and cy2 > cy1:
        cropped = image[cy1:cy2, cx1:cx2]
        return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return cv2.resize(image, target_size)

# =============================================================================
# 3. HELPER FUNCTIONS: OCR EXTRACTION
# =============================================================================

def extract_data_from_cell(crop_img, method):
    """Applies specific OCR logic based on the method tag."""
    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    if method == 'default':
        return pytesseract.image_to_string(rgb).strip()
    
    elif method == 'digits_only':
        # Allowed: Digits and decimal points
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
        return pytesseract.image_to_string(rgb, config=config).strip()
    
    elif method == 'signature_check':
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        return "SIGNED" if cv2.countNonZero(bin_img) > 500 else "NOT SIGNED"
    
    return ""

# =============================================================================
# 4. PIPELINE STAGES
# =============================================================================

def run_stage_1_preprocessing():
    """Converts TIFF -> Deskewed/Cropped PNG."""
    print("\n--- STAGE 1: Pre-processing (TIFF to PNG) ---")
    os.makedirs(INTERMEDIATE_PNG_FOLDER, exist_ok=True)
    
    files = glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.tif*')) + \
            glob.glob(os.path.join(INPUT_TIFF_FOLDER, '*.TIF*'))
    
    if not files:
        print("No TIFF files found.")
        return

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        print(f"Processing {i+1}/{len(files)}: {fname}")
        
        img = cv2.imread(fpath) # Reads page 1 by default
        if img is None: continue
            
        deskewed = deskew_single_image(img)
        final_img = find_anchors_and_crop(deskewed)
        
        save_name = os.path.splitext(fname)[0] + ".png"
        cv2.imwrite(os.path.join(INTERMEDIATE_PNG_FOLDER, save_name), final_img)

def run_stage_2_extraction():
    """Converts PNG -> 8 Cell Crops -> Excel Report."""
    print("\n--- STAGE 2: Extraction (PNG to Data) ---")
    os.makedirs(FINAL_OUTPUT_FOLDER, exist_ok=True)
    
    files = glob.glob(os.path.join(INTERMEDIATE_PNG_FOLDER, '*.png'))
    if not files:
        print("No PNG files found to extract.")
        return

    all_data = []
    
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        base_name = os.path.splitext(fname)[0]
        print(f"Extracting {i+1}/{len(files)}: {fname}")
        
        full_img = cv2.imread(fpath)
        if full_img is None: continue
            
        # Setup subfolder for cell images
        subfolder = os.path.join(FINAL_OUTPUT_FOLDER, base_name)
        os.makedirs(subfolder, exist_ok=True)
        
        row_data = {'Filename': fname}
        
        # Process 8 Cells
        for key, conf in CELL_CONFIG.items():
            x1, y1, x2, y2 = conf['coords']
            method = conf['method']
            
            # Safety crop
            h, w = full_img.shape[:2]
            crop = full_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            # Save Crop
            cv2.imwrite(os.path.join(subfolder, f"{key}.png"), crop)
            
            # Extract
            text = extract_data_from_cell(crop, method)
            row_data[key] = text
            
        all_data.append(row_data)
        
    # Save Excel
    if all_data:
        df = pd.DataFrame(all_data)
        out_path = os.path.join(FINAL_OUTPUT_FOLDER, 'Final_Report.xlsx')
        df.to_excel(out_path, index=False)
        print(f"\nSUCCESS! Report saved to: {out_path}")
        print(df.head())

# =============================================================================
# 5. MAIN EXECUTION
# =============================================================================

def main():
    print("=== STARTING AUTOMATED PIPELINE ===")
    
    # 1. Run TIFF -> PNG
    run_stage_1_preprocessing()
    
    # 2. Run PNG -> Data
    run_stage_2_extraction()
    
    print("\n=== PIPELINE COMPLETE ===")

# Run the main function
if __name__ == "__main__":
    main()
