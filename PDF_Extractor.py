def detect_form_with_llm(image):
    """
    Crops the header and uses the LLM to decide if it's CMS 1500 or 1450.
    """
    h, w = image.shape[:2]
    
    # 1. Crop the top 30% (Header area)
    # This contains the QR code AND the Form Title text
    header_crop = image[0:int(h*0.3), 0:w]
    
    # 2. Define strict instruction for the LLM
    instruction = (
        "Analyze this document header. Determine the Form Type. "
        "Rules: "
        "1. If you see 'HEALTH INSURANCE CLAIM FORM' or a QR code on the left, it is 'CMS 1500'. "
        "2. If you see 'UB-04' or 'CMS 1450', it is 'CMS 1450'. "
        "Return the result as a single key JSON: {\"Type\": \"CMS 1500\"} or {\"Type\": \"CMS 1450\"}"
    )
    
    # 3. Call your existing smart agent
    # We pass an empty context string "" because we don't need correction, just classification
    try:
        response = call_smart_agent(header_crop, instruction, "")
        
        # Handle cases where response might be a string or dict
        # Assuming call_smart_agent returns a dictionary like row.update(data) expects
        if isinstance(response, dict):
            return response.get("Type", "CMS 1500") # Default to 1500 if key missing
        else:
            # If it returned a string, sanitize it
            text = str(response).upper()
            if "1450" in text or "UB" in text:
                return "CMS 1450"
            return "CMS 1500"
            
    except Exception as e:
        print(f"LLM Classification failed: {e}. Defaulting to CMS 1500.")
        return "CMS 1500"

import os
import cv2
import numpy as np
import glob
import json
import base64
import pandas as pd
import shutil
import time
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.styles import Alignment
from openai import AzureOpenAI, OpenAI

# ==========================================
# 1. USER CONFIGURATION (EDIT PATHS HERE)
# ==========================================

# !!! ENTER YOUR API KEY HERE !!!
API_KEY = "YOUR_OPENAI_API_KEY_HERE"

# Client Initialization (Uncomment the one you use)
# client = OpenAI(api_key=API_KEY)
# client = AzureOpenAI(api_key=API_KEY, api_version="2024-02-01", azure_endpoint="YOUR_ENDPOINT")

# !!! UPDATE YOUR FOLDER PATH HERE !!!
BASE_FOLDER = r'C:\CMS_Project' 

INPUT_FOLDER = os.path.join(BASE_FOLDER, 'Input_Raw')
INTERMEDIATE_FOLDER = os.path.join(BASE_FOLDER, 'Intermediate_Processed')
FINAL_EXCEL_FOLDER = os.path.join(BASE_FOLDER, 'Final_Reports')

BATCH_SIZE = 100 
TARGET_SIZE = (2480, 3508) # Standard CMS1500 dimensions

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
        5. "ZZ" vs "22": In ID columns, 'ZZ' is a qualifier, '22' might be a Place of Service code.
        6. NPI check: NPI is ALWAYS a 10-digit number.
        """,
        "Regions": {
            # --- REGION 1: Bottom Blocks (31, 32, 33) ---
            "Region_1_Bottom": [0, 0, 0, 0, # <--- UPDATE COORDS (x1, y1, x2, y2)
                                """
                                Extract the following blocks:
                                1. Block 31: Signed? (Yes/No), Signature Name (Text).
                                2. Block 32: Facility Name, 32a (NPI - 10 digits), 32b (Alphanumeric/Numeric).
                                3. Block 33: Billing Provider Name, 33a (NPI), 33b (Alphanumeric).
                                
                                Return JSON Keys: 
                                B31_Is_Signed, B31_Sign_Name, B32_Facility_Name, B32a_NPI, B32b_OtherID, B33_Billing_Name, B33a_NPI, B33b_OtherID
                                """],
            
            # --- REGION 2: Service Lines (Block 24 I & J) ---
            "Region_2_Services": [0, 0, 0, 0, # <--- UPDATE COORDS (x1, y1, x2, y2)
                                  """
                                  Focus on Block 24, Columns I (Qualifier) and J (Rendering Provider ID).
                                  - There may be multiple rows.
                                  - Data is often stacked in Col J (e.g., top is alphanumeric, bottom is NPI).
                                  - Capture PAIRS of (Indicator, Value).
                                  
                                  Example Output Format:
                                  "Row1": "(ZZ, NPI): (AB123, 1234567890)"
                                  "Row2": "(G2, NPI): (XY999, 9876543210)"
                                  
                                  Return JSON Keys:
                                  B24_Row1_Data, B24_Row2_Data, B24_Row3_Data, B24_Row4_Data, B24_Row5_Data, B24_Row6_Data
                                  """]
        }
    },
    
    "CMS1450": {
        "Excel_Filename": "CMS1450_UB04_Extraction_Report.xlsx",
        "Correction_Context": """
        CRITICAL OCR CORRECTION RULES:
        1. NPI Numbers are 10 digits.
        2. TIN/EIN is usually 9 digits.
        3. Qualifiers are usually 2 letters (e.g., 1G, G2, 0B).
        4. Do not confuse 'O' (Letter) with '0' (Zero).
        """,
        "Regions": {
            # --- REGION 1: Top Left (Provider/Pay-to/Tax) ---
            "Region_1_Header": [0, 0, 0, 0, # <--- UPDATE COORDS
                                """
                                Extract:
                                1. Block 1: Provider Name (Name only).
                                2. Block 2: Pay-to Address (Full address).
                                3. Block 5: Fed Tax No (TIN/EIN).
                                
                                Return JSON Keys:
                                B1_Provider_Name, B2_PayTo_Address, B5_TaxID
                                """],

            # --- REGION 2: Middle Right (NPIs) ---
            "Region_2_NPIs": [0, 0, 0, 0, # <--- UPDATE COORDS
                              """
                              Extract:
                              1. Block 56: NPI (Billing Provider).
                              2. Block 57: Other Provider ID.
                              
                              Return JSON Keys:
                              B56_Billing_NPI, B57_Other_ID
                              """],

            # --- REGION 3: Attending/Operating (Blocks 76-79) ---
            "Region_3_Providers": [0, 0, 0, 0, # <--- UPDATE COORDS
                                   """
                                   For Blocks 76, 77, 78, and 79, extract the sub-fields:
                                   - a (NPI)
                                   - b (Qual)
                                   - c (Last Name)
                                   - d (First Name)
                                   
                                   Return JSON Keys:
                                   B76_Attending_NPI, B76_Qual, B76_Last, B76_First,
                                   B77_Operating_NPI, B77_Qual, B77_Last, B77_First,
                                   B78_Other_NPI, B78_Qual, B78_Last, B78_First,
                                   B79_Other_NPI, B79_Qual, B79_Last, B79_First
                                   """],

            # --- REGION 4: Block 81 (Code-Code) ---
            "Region_4_Codes": [0, 0, 0, 0, # <--- UPDATE COORDS
                               """
                               Extract Block 81 (Code-Code Field). 
                               There are 4 rows (a, b, c, d). Extract all text from each line.
                               
                               Return JSON Keys:
                               B81a_Details, B81b_Details, B81c_Details, B81d_Details
                               """]
        }
    }
}

# ==========================================
# 3. COMPLETE PRE-PROCESSING FUNCTIONS
# ==========================================

def deskew_single_image(image):
    """
    Deskews an image using projection profiles.
    Crucial for aligning the grid lines.
    """
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
    # print(f"   -> Deskewing by {best_angle} degrees")
    
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def detect_form_type_by_qr(image):
    """
    Analyzes the Top-Left of the image (first 1000px) for a QR Code.
    - Found = CMS1500
    - Not Found = CMS1450 (UB04)
    """
    # Look at a generous top-left crop to catch the QR even if margins vary
    h, w = image.shape[:2]
    crop_size = min(1000, w, h)
    tl_crop = image[0:crop_size, 0:crop_size]
    
    # 1. Standard QR Detection
    detector = cv2.QRCodeDetector()
    retval, decoded_info, points, straight_qrcode = detector.detectAndDecodeMulti(tl_crop)
    
    if retval:
        return "CMS1500"
        
    # 2. Heuristic: Look for dense black square (Fall back if QR is blurry)
    gray = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in cnts:
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = float(cw) / ch
        # QRs are roughly square and usually > 50px
        if 50 < cw < 500 and 50 < ch < 500 and 0.8 < aspect < 1.2:
            roi = thresh[y:y+ch, x:x+cw]
            density = cv2.countNonZero(roi) / (cw * ch)
            # High density of black pixels = likely a QR/DataMatrix
            if density > 0.4: 
                return "CMS1500"
                
    return "CMS1450"

def detect_and_crop_grid(image):
    """
    Detects the main table grid and crops to TARGET_SIZE.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Horizontal & Vertical Lines
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
        
        # Validation: Grid must be >10% of image
        if (cw * ch) > (0.1 * w * h): 
            pad = 20
            # Safe Crop
            y1, y2 = max(0, y-pad), min(h, y+ch+pad)
            x1, x2 = max(0, x-pad), min(w, x+cw+pad)
            
            crop = image[y1:y2, x1:x2]
            return True, cv2.resize(crop, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
    return False, None

def find_anchors_and_crop(image):
    """Fallback Cropping logic if Grid fails."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    
    dilated = cv2.dilate(thresh, np.ones((10,10), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts: return cv2.resize(image, TARGET_SIZE)
    
    # Top-Left (Smallest x+y) & Bottom-Right (Largest x+y+w+h)
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
    """Calls OpenAI GPT-4o with image and instructions."""
    if crop_img is None: return {}
    b64_img = encode_image(crop_img)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": f"Medical OCR Agent. OUTPUT JSON ONLY.\nCONTEXT & ERROR CHECKING:\n{context}"},
                {"role": "user", "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"}}
                ]}
            ],
            temperature=0, 
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        return json.loads(content)
    except Exception as e:
        print(f"   -> LLM Error: {e}")
        return {"Error": "Agent_Failed"}

def save_excel(data, filepath):
    """Saves data to Excel with formatting."""
    if not data: return
    df = pd.DataFrame(data)
    
    # Move Filename to front
    cols = list(df.columns)
    if 'Filename' in cols: cols.insert(0, cols.pop(cols.index('Filename')))
    df = df[cols]
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
        
    wb = load_workbook(filepath)
    ws = wb['Data']
    
    # Create Table
    tab = Table(displayName="SmartTable", ref=ws.dimensions)
    tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
    ws.add_table(tab)
    
    # Format Columns
    for col in ws.columns:
        length = max(len(str(c.value)) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(length + 2, 50)
        for c in col: 
            c.alignment = Alignment(wrap_text=True, vertical='top')
        
    wb.save(filepath)

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    print("--- STARTING MEDICAL FORM EXTRACTOR ---")
    os.makedirs(INTERMEDIATE_FOLDER, exist_ok=True)
    os.makedirs(FINAL_EXCEL_FOLDER, exist_ok=True)
    
    files = glob.glob(os.path.join(INPUT_FOLDER, '*.*'))
    print(f"Found {len(files)} files in Input folder.")
    
    # PROCESS IN BATCHES
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1} ({len(batch)} files) ---")
        
        results = {k: [] for k in SMART_CONFIG.keys()}
        
        for fpath in batch:
            original_fname = os.path.basename(fpath)
            print(f"Reading: {original_fname}...")
            
            img = cv2.imread(fpath)
            if img is None: continue
            
            # --- STEP 1: DESKEW (Align Rotation) ---
            deskewed_img = deskew_single_image(img)
            
            # --- STEP 2: DETECT TYPE BY QR (Top Left) ---
            # We check the deskewed image so coordinates are reliable
            active_type = detect_form_type_by_qr(deskewed_img)
            print(f"   -> Classified as: {active_type}")
            
            # --- STEP 3: CROP TO GRID (Normalize to 2480x3508) ---
            is_grid, processed_img = detect_and_crop_grid(deskewed_img)
            if not is_grid:
                # Fallback if grid detection fails
                processed_img = find_anchors_and_crop(deskewed_img)
            
            # --- STEP 4: SAVE INTERMEDIATE (With Classification Name) ---
            new_fname = f"{active_type}_{original_fname}"
            if not new_fname.endswith(".png"): new_fname += ".png"
            
            save_path = os.path.join(INTERMEDIATE_FOLDER, new_fname)
            cv2.imwrite(save_path, processed_img)
            
            # --- STEP 5: EXTRACT DATA ---
            row = {"Filename": new_fname, "Form_Type": active_type}
            cfg = SMART_CONFIG[active_type]
            
            for r_key, params in cfg['Regions'].items():
                x1, y1, x2, y2 = params[0], params[1], params[2], params[3]
                instruction = params[4]
                
                # Skip if coordinates are not set yet (0,0,0,0)
                if x1 == 0 and x2 == 0:
                    continue
                    
                # CROP REGION
                crop = processed_img[int(y1):int(y2), int(x1):int(x2)]
                
                # CALL AI
                data = call_smart_agent(crop, instruction, cfg['Correction_Context'])
                row.update(data)
                
            results[active_type].append(row)
            
        # --- STEP 6: SAVE EXCEL BATCH ---
        timestamp = int(time.time())
        
        for f_type, rows in results.items():
            if rows:
                base_name = SMART_CONFIG[f_type]['Excel_Filename']
                # Create unique batch name
                fname = base_name.replace(".xlsx", f"_Batch_{timestamp}.xlsx")
                save_excel(rows, os.path.join(FINAL_EXCEL_FOLDER, fname))
                print(f"Saved Batch Report: {fname}")
                
        print("Batch Memory Cleared.")

if __name__ == "__main__":
    main()
