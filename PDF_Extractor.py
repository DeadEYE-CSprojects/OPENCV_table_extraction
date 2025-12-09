import os
import cv2
import numpy as np
import glob
import json
import base64
import pandas as pd
import shutil
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Alignment
from openai import AzureOpenAI, OpenAI

# Try importing pdf2image for PDF support
try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: 'pdf2image' not installed. PDF files will be skipped.")

# ==========================================
# 1. USER CONFIGURATION
# ==========================================

# !!! ENTER YOUR API KEY HERE !!!
API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# Client Initialization (Uncomment the one you use)
# client = OpenAI(api_key=API_KEY)
client = AzureOpenAI(api_key=API_KEY, api_version="2024-02-01", azure_endpoint="YOUR_ENDPOINT")

# PATHS
BASE_FOLDER = r'C:\CMS_Project' 

FOLDER_1_PNG = os.path.join(BASE_FOLDER, '1_Standardized_PNG')
FOLDER_2_FORMS = os.path.join(BASE_FOLDER, '2_Aligned_Forms')
FOLDER_3_REGIONS = os.path.join(BASE_FOLDER, '3_Region_Crops')
FOLDER_4_REPORT = os.path.join(BASE_FOLDER, '4_Final_Reports')

BATCH_SIZE = 100 
TARGET_SIZE = (2480, 3508) # Standard CMS1500 dimensions (A4 @ 300DPI approx)

# ==========================================
# 2. SMART CONFIGURATION (THE BRAIN)
# ==========================================

SMART_CONFIG = {
    "CMS1500": {
        "Excel_Filename": "CMS1500_Extraction_Report.xlsx",
        "Correction_Context": """
        CRITICAL OCR CORRECTION RULES:
        1. "0" (Zero) vs "U": If a zero is cut by grid lines, it looks like 'U'. Context is usually numeric.
        2. "4" vs "1": Watch for '4' being misread as '1'.
        3. "8" vs "2" or "B": '8' is often misread as '2', 'B', or '3'.
        4. "5" vs "6": Differentiate carefully.
        5. NPI check: NPI is ALWAYS a 10-digit number.
        """,
        "Regions": {
            # Format: [x1, y1, x2, y2, "Instruction"]
            "Region_1_Bottom": [0, 0, 0, 0, 
                                """
                                Extract Block 31 (Signature), Block 32 (Facility), Block 33 (Billing Provider).
                                Return JSON Keys: B31_Sign_Name, B32_Facility_Name, B33_Billing_Name
                                """],
            "Region_2_Services": [0, 0, 0, 0, 
                                  """
                                  Extract Block 24 (Service Lines). Return JSON Keys: B24_Row1_Data, B24_Row2_Data
                                  """]
        }
    },
    
    "CMS1450": {
        "Excel_Filename": "CMS1450_UB04_Extraction_Report.xlsx",
        "Correction_Context": """
        CRITICAL OCR CORRECTION RULES:
        1. NPI Numbers are 10 digits.
        2. TIN/EIN is usually 9 digits.
        """,
        "Regions": {
            "Region_1_Header": [0, 0, 0, 0, 
                                """
                                Extract Block 1 (Name), Block 2 (Pay-to), Block 5 (Tax ID).
                                Return JSON Keys: B1_Name, B2_PayTo, B5_TaxID
                                """],
            "Region_2_NPIs": [0, 0, 0, 0, 
                              """
                              Extract Block 56 (NPI) and 57 (Other ID).
                              Return JSON Keys: B56_NPI, B57_OtherID
                              """]
        }
    }
}

# ==========================================
# 3. HELPER FUNCTIONS (IMAGE & UTILS)
# ==========================================

def convert_to_opencv_image(filepath):
    """Converts PDF or Image file to OpenCV format (numpy array)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.pdf':
        if not PDF_SUPPORT: return None
        try:
            # Convert first page only
            pages = convert_from_path(filepath, first_page=1, last_page=1, dpi=300)
            if pages:
                return cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PDF Error {filepath}: {e}")
            return None

    elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        return cv2.imread(filepath)
    
    return None

def deskew_single_image(image):
    """Corrects image rotation (skew)."""
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
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(thresh, M_rot, (w, h), flags=cv2.INTER_NEAREST)
        score = np.var(np.sum(rotated, axis=1))
        scores.append(score)
    
    best_angle = angles[np.argmax(scores)]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def detect_form_type_by_qr(image):
    """Uses LLM to detect if form is CMS1500 or CMS1450."""
    h, w = image.shape[:2]
    header_crop = image[0:int(h*0.3), 0:w]
    
    instruction = (
        "Analyze this document header. Determine the Form Type. "
        "Rules: "
        "1. If 'HEALTH INSURANCE CLAIM FORM' or QR code on left -> 'CMS1500'. "
        "2. If 'UB-04' or 'CMS 1450' -> 'CMS1450'. "
        "Return JSON: {\"Type\": \"CMS1500\"} or {\"Type\": \"CMS1450\"}"
    )
    
    try:
        response = call_smart_agent(header_crop, instruction, "")
        if isinstance(response, dict):
            return response.get("Type", "CMS1500")
        return "CMS1500" # Default
    except:
        return "CMS1500"

def detect_and_crop_grid(image):
    """Detects the main table grid lines and crops to them."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 38, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 380))
    
    img_hor = cv2.dilate(cv2.erode(thresh, hor_kernel), hor_kernel)
    img_ver = cv2.dilate(cv2.erode(thresh, ver_kernel), ver_kernel)
    
    combined = cv2.add(img_hor, img_ver)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        if (cw * ch) > (0.1 * w * h): 
            pad = 20
            y1, y2 = max(0, y-pad), min(h, y+ch+pad)
            x1, x2 = max(0, x-pad), min(w, x+cw+pad)
            crop = image[y1:y2, x1:x2]
            return True, cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    return False, None

def find_anchors_and_crop(image):
    """Fallback cropping if grid detection fails."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    dilated = cv2.dilate(thresh, np.ones((10,10), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: return cv2.resize(image, TARGET_SIZE)
    
    tl_c = min(cnts, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1])
    br_c = max(cnts, key=lambda c: cv2.boundingRect(c)[0] + cv2.boundingRect(c)[1] + cv2.boundingRect(c)[2] + cv2.boundingRect(c)[3])
    
    x1, y1, _, _ = cv2.boundingRect(tl_c)
    bx, by, bw, bh = cv2.boundingRect(br_c)
    x2, y2 = bx+bw, by+bh
    
    pad = 20
    crop = image[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    return cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)

# ==========================================
# 4. LLM & EXCEL FUNCTIONS
# ==========================================

def encode_image(image_arr):
    _, buffer = cv2.imencode('.png', image_arr)
    return base64.b64encode(buffer).decode('utf-8')

def call_smart_agent(crop_img, instruction, context):
    """Sends image crop to OpenAI."""
    if crop_img is None: return {}
    
    # Upscale very small crops for better OCR
    h,w = crop_img.shape[:2]
    if h < 200:
        crop_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    b64_img = encode_image(crop_img)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Medical OCR Agent. OUTPUT JSON ONLY.\n{context}"},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                ]}
            ],
            temperature=0, 
            max_tokens=800
        )
        content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"   -> AI Error: {e}")
        return {}

def save_excel(data, filepath):
    """Saves list of dicts to Excel with appending support."""
    if not data: return
    df = pd.DataFrame(data)
    
    # Ensure Filename is first column
    cols = list(df.columns)
    if 'Filename' in cols: cols.insert(0, cols.pop(cols.index('Filename')))
    df = df[cols]
    
    if not os.path.exists(filepath):
        df.to_excel(filepath, index=False)
    else:
        with pd.ExcelWriter(filepath, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            writer.workbook = load_workbook(filepath)
            start_row = writer.workbook.active.max_row
            df.to_excel(writer, index=False, header=False, startrow=start_row)

    # Styling
    wb = load_workbook(filepath)
    ws = wb.active
    tab = Table(displayName="Data", ref=ws.dimensions)
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
    try: ws.add_table(tab)
    except: pass # Table might already exist
    
    wb.save(filepath)

# ==========================================
# PHASE 1: STANDARDIZE TO PNG
# ==========================================
def run_phase_1_conversion(input_folder):
    print("\n--- PHASE 1: CONVERTING RAW FILES TO PNG ---")
    os.makedirs(FOLDER_1_PNG, exist_ok=True)
    
    files = glob.glob(os.path.join(input_folder, '*.*'))
    count = 0
    
    for fpath in files:
        fname = os.path.basename(fpath)
        dest_path = os.path.join(FOLDER_1_PNG, os.path.splitext(fname)[0] + ".png")
        
        if os.path.exists(dest_path): continue

        img = convert_to_opencv_image(fpath)
        if img is not None:
            cv2.imwrite(dest_path, img)
            count += 1
            if count % 10 == 0: print(f"Converted {count} files...")
    
    print(f"Phase 1 Complete. {count} new files created.")

# ==========================================
# PHASE 2: ALIGN & CLASSIFY
# ==========================================
def run_phase_2_preprocessing():
    print("\n--- PHASE 2: DESKEW, CROP & CLASSIFY ---")
    os.makedirs(FOLDER_2_FORMS, exist_ok=True)
    
    files = glob.glob(os.path.join(FOLDER_1_PNG, '*.png'))
    
    for fpath in files:
        fname = os.path.basename(fpath)
        
        # NOTE: We can't check 'if exists' simply by name because output name changes based on Type.
        # But we can check if any file ending in _{fname} exists in Folder 2.
        
        img = cv2.imread(fpath)
        if img is None: continue

        # 1. Deskew
        deskewed = deskew_single_image(img)
        
        # 2. Detect Type (Needed for filename)
        form_type = detect_form_type_by_qr(deskewed)
        
        # 3. Crop to Grid
        is_grid, processed = detect_and_crop_grid(deskewed)
        if not is_grid:
            processed = find_anchors_and_crop(deskewed)

        # 4. Save: Format = "Type_OriginalName.png"
        new_name = f"{form_type}_{fname}"
        save_path = os.path.join(FOLDER_2_FORMS, new_name)
        
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, processed)
            print(f"Processed: {new_name}")

# ==========================================
# PHASE 3: CROP REGIONS
# ==========================================
def run_phase_3_region_cropping():
    print("\n--- PHASE 3: CUTTING REGIONS TO FOLDERS ---")
    
    files = glob.glob(os.path.join(FOLDER_2_FORMS, '*.png'))
    
    for fpath in files:
        fname = os.path.basename(fpath)
        
        # Parse Type from Filename (e.g., "CMS1500_File1.png")
        parts = fname.split('_', 1)
        if len(parts) < 2: continue # Safety check
        
        form_type = parts[0] # "CMS1500"
        original_name = parts[1] # "File1.png"
        clean_name = os.path.splitext(original_name)[0]
        
        # FOLDER NAME FORMAT: "CMS1500_File1"
        # This contains BOTH Type and Name, easy for Phase 4.
        folder_name = f"{form_type}_{clean_name}"
        file_crop_dir = os.path.join(FOLDER_3_REGIONS, folder_name)
        os.makedirs(file_crop_dir, exist_ok=True)

        if form_type not in SMART_CONFIG: continue

        img = cv2.imread(fpath)
        h, w = img.shape[:2]
        cfg = SMART_CONFIG[form_type]
        PAD = 20

        # Crop all regions in config
        for r_key, params in cfg['Regions'].items():
            x1, y1, x2, y2, _ = params
            
            if x1 == 0 and x2 == 0: continue

            y1_pad, y2_pad = max(0, int(y1)-PAD), min(h, int(y2)+PAD)
            x1_pad, x2_pad = max(0, int(x1)-PAD), min(w, int(x2)+PAD)

            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Save: "Region_Name.png" inside the specific file folder
            cv2.imwrite(os.path.join(file_crop_dir, f"{r_key}.png"), crop)
        
        print(f"Crops ready for: {folder_name}")

# ==========================================
# PHASE 4: AI EXTRACTION
# ==========================================
def run_phase_4_extraction():
    print("\n--- PHASE 4: AI EXTRACTION ---")
    os.makedirs(FOLDER_4_REPORT, exist_ok=True)
    
    # Get all sub-folders in Folder 3
    doc_folders = glob.glob(os.path.join(FOLDER_3_REGIONS, '*'))
    
    results = {k: [] for k in SMART_CONFIG.keys()}
    
    for doc_dir in doc_folders:
        if not os.path.isdir(doc_dir): continue
        
        folder_name = os.path.basename(doc_dir) # e.g., "CMS1500_File1"
        print(f"Analyzing: {folder_name}...")
        
        # 1. DETECT FORM TYPE FROM FOLDER NAME
        # Splitting "CMS1500_File1" -> ["CMS1500", "File1"]
        parts = folder_name.split('_', 1)
        if len(parts) < 2: 
            print("   -> Skipping: Invalid Folder Name Format")
            continue
            
        form_type = parts[0]
        actual_filename = parts[1]

        if form_type not in SMART_CONFIG:
            print(f"   -> Skipping: Unknown Type {form_type}")
            continue

        # 2. Setup Data Row
        row_data = {"Filename": actual_filename, "Form_Type": form_type}
        cfg = SMART_CONFIG[form_type]
        
        # 3. Process Each Region defined in Config
        for r_key, params in cfg['Regions'].items():
            instruction = params[4]
            context = cfg['Correction_Context']
            
            # Look for the specific image inside the folder
            crop_path = os.path.join(doc_dir, f"{r_key}.png")
            
            if os.path.exists(crop_path):
                # We found the pre-cut image!
                crop_img = cv2.imread(crop_path)
                
                # Send to AI
                extracted_data = call_smart_agent(crop_img, instruction, context)
                row_data.update(extracted_data)
            else:
                # Region image missing (maybe coords were 0 or crop failed)
                pass

        results[form_type].append(row_data)

    # 4. Save Final Reports
    for f_type, rows in results.items():
        if rows:
            fname = SMART_CONFIG[f_type]['Excel_Filename']
            save_path = os.path.join(FOLDER_4_REPORT, fname)
            save_excel(rows, save_path)
            print(f"SAVED: {len(rows)} rows to {fname}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # Run the Pipeline Phases Step-by-Step
    # Comment out phases you have already finished!

    # 1. Convert everything to PNG
    run_phase_1_conversion(os.path.join(BASE_FOLDER, 'Input_Raw'))

    # 2. Deskew, Classify, and Align
    run_phase_2_preprocessing()

    # 3. Cut Region Images into Folders (e.g. "CMS1500_FileA")
    run_phase_3_region_cropping()

    # 4. Send Crops to AI and Build Excel
    run_phase_4_extraction()
