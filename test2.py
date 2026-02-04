import os
import shutil
import pandas as pd
import re
import cv2
import numpy as np
import pytesseract
import subprocess
import json
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from openai import OpenAI
import openpyxl
from docx2pdf import convert as docx_to_pdf_convert
from PIL import Image
import time
import base64
import argparse

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================

# --- CONTROL SWITCH ---
# Set to FALSE: The script will ONLY convert files to .txt and stop.
# Set to TRUE: The script will run the 'contract_type.py' analysis after conversion.
RUN_CONTRACT_SCRIPT = False

# API SETUP
# Replace "YOUR_OPENAI_API_KEY_HERE" with your actual key if not using env variables
API_KEY = "YOUR_OPENAI_API_KEY_HERE"
client = OpenAI(api_key=API_KEY)

# IMAGE QUALITY SETTINGS
# 500 DPI is significantly higher than standard (200) to ensure small text is clear for OCR.
PDF_CONVERSION_DPI = 500      
# If an image width is below 2500px, we will upscale it before sending to LLM.
MIN_IMAGE_WIDTH = 2500        

# DIRECTORY PATHS
INPUT_FILES_PATH = "./input_files/"
TXT_OUTPUT_PATH = "./txt_output/"       # Folder where finalized .txt files are saved
FUTURE_EXCEL_PATH = "./final_output/"   # Reserved folder for Phase 2 output
CONTRACT_SCRIPTS_PATH = "./contract_scripts/"

# FILE NAMES
INVENTORY_FILE_PATH = "inventory.xlsx"  # The Master List for parallel processing
TOKEN_CALC_PATH = "token_calculation.xlsx"
PROCESS_LOG_PATH = "process_log.txt"

# KNOWN CONTRACT TYPES (Used for LLM Classification)
CONTRACT_TYPES_LIST = [
    "home_health", "skilled_nursing", "aec", "asc", "detox", "dialysis", 
    "hca", "hospice", "psych", "rehab", "tenet", "prosthetics", "drg", 
    "cah", "rhc", "dme", "anesthesia", "chv", "surgery", "telemedicine", "audiology"
]

# TESSERACT CONFIG
# If you are on Windows, you might need to uncomment and set the path below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# CREATE DIRECTORIES IF THEY DO NOT EXIST
os.makedirs(INPUT_FILES_PATH, exist_ok=True)
os.makedirs(TXT_OUTPUT_PATH, exist_ok=True)
os.makedirs(CONTRACT_SCRIPTS_PATH, exist_ok=True)
os.makedirs(FUTURE_EXCEL_PATH, exist_ok=True)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_cis_id(filename):
    """
    Extracts the first complete numeric string of at least 4 digits from the filename.
    Returns "UNKNOWN" if no ID is found.
    """
    match = re.search(r'\d{4,}', filename)
    return match.group(0) if match else "UNKNOWN"

def deskew_image(pil_image):
    """
    Corrects slight tilts in images (0.1 - 1.0 degrees) which can confuse OCR.
    Uses OpenCV to find text contours and rotate the image to be perfectly horizontal.
    """
    img = np.array(pil_image)
    
    # Convert image to BGR format for OpenCV
    if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale and invert (text becomes white, background black)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    
    # Threshold to isolate text pixels
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    
    # If no text found, return original
    if len(coords) == 0: return pil_image

    # Calculate the angle of the text block
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle

    # Rotate the image to correct the skew
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))

def enhance_and_upscale(pil_image):
    """
    Prepares an image for High-Accuracy OCR.
    1. Checks if resolution is too low (< 2500px width). If so, upscales it.
    2. Applies a sharpening filter to make text edges crisp.
    """
    img = np.array(pil_image)
    
    # 1. Smart Upscale using Cubic Interpolation
    height, width = img.shape[:2]
    if width < MIN_IMAGE_WIDTH:
        scale_factor = MIN_IMAGE_WIDTH / width
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # 2. Apply Sharpening Kernel
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced = cv2.filter2D(img, -1, kernel)
    
    return Image.fromarray(enhanced)

def encode_image_base64(image_path):
    """
    Reads an image file from disk and returns a Base64 encoded string.
    Required for sending images to OpenAI API.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def check_page_complexity(image_path):
    """
    Uses GPT-4o Vision to visually inspect the page.
    Returns True if it contains complex elements (Tables, Forms, Diagrams).
    Also returns the token cost of this check.
    """
    base64_img = encode_image_base64(image_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Does this page contain a data table, form grid, or significant diagram? Answer YES or NO."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                ]
            }],
            max_tokens=15
        )
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer, response.usage.total_tokens
    except Exception as e:
        print(f"      [Warning] Complexity Check Failed: {e}")
        return False, 0

def llm_convert_to_text(pil_image):
    """
    Transcribes a High-Resolution image using GPT-4o.
    Saves image as PNG (Lossless) first to ensure no compression artifacts.
    """
    # Save as PNG to preserve the 500 DPI quality we generated earlier
    temp_path = "temp_llm_input.png"
    pil_image.save(temp_path, format="PNG") 
    
    base64_img = encode_image_base64(temp_path)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert OCR engine. Transcribe the text from this high-resolution image exactly. Preserve all table structures using markdown. Do not summarize."},
                {"role": "user", "content": [{"type": "text", "text": "Transcribe this image."}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"}}]}
            ],
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        # Cleanup temp file
        if os.path.exists(temp_path): os.remove(temp_path)
        return content, tokens
    except Exception as e:
        print(f"      [Error] LLM Conversion Failed: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return "", 0

def log_token_usage_excel(filename, phase1, phase2=0):
    """
    Appends token usage row-by-row to the Excel file.
    This ensures that if the script crashes, we don't lose the cost data for previous files.
    """
    new_data = pd.DataFrame([[filename, phase1, phase2]], columns=["filename", "phase1_token", "phase2_tokens"])
    try:
        if not os.path.exists(TOKEN_CALC_PATH):
            new_data.to_excel(TOKEN_CALC_PATH, index=False)
        else:
            # Append to existing Excel sheet
            with pd.ExcelWriter(TOKEN_CALC_PATH, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                if writer.book.worksheets:
                    start_row = writer.book.active.max_row
                    new_data.to_excel(writer, index=False, header=False, startrow=start_row)
                else:
                    new_data.to_excel(writer, index=False)
    except Exception as e:
        print(f"      [Warning] Could not log tokens: {e}")

def determine_contract_type(text_content):
    """
    Analyzes the text content (First 4k chars + Last 2k chars) to identify the contract type.
    """
    context = text_content[:4000] + "\n...\n" + text_content[-2000:]
    prompt = f"Analyze and classify into one of: {json.dumps(CONTRACT_TYPES_LIST)}. If unsure, return 'others'."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Return only the category name."},
                {"role": "user", "content": f"{prompt}\n\nTEXT:\n{context}"}
            ]
        )
        ctype = response.choices[0].message.content.strip().lower()
        # Verify result is in our allowed list
        for t in CONTRACT_TYPES_LIST:
            if t in ctype: return t
        return "others"
    except:
        return "others"

def log_process_status(message):
    """Writes a simple log message to a text file with a timestamp."""
    with open(PROCESS_LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# ==========================================
# 3. MAIN WORKFLOW
# ==========================================

def get_or_create_inventory():
    """
    LOGIC FOR PARALLEL PROCESSING:
    1. Check if 'inventory.xlsx' ALREADY exists.
       - If YES: Load it strictly. DO NOT create a new one. This ensures all team members use the exact same Index list.
    2. If NO: Scan input folder, create the Excel file, save it, and return it.
    """
    if os.path.exists(INVENTORY_FILE_PATH):
        print(f"--- LOCKED: Found existing 'inventory.xlsx'. Loading Master List... ---")
        return pd.read_excel(INVENTORY_FILE_PATH)
    else:
        print(f"--- INITIALIZING: Creating NEW 'inventory.xlsx' from {INPUT_FILES_PATH} ---")
        if not os.path.exists(INPUT_FILES_PATH):
            print(f"CRITICAL ERROR: Input directory {INPUT_FILES_PATH} missing.")
            return pd.DataFrame() # Return empty DF

        input_files = [f for f in os.listdir(INPUT_FILES_PATH) if os.path.isfile(os.path.join(INPUT_FILES_PATH, f))]
        
        # Sort files to ensure the initial creation is deterministic (alphabetical order)
        input_files.sort()

        inventory_data = []
        for idx, f in enumerate(input_files):
            inventory_data.append({
                "Index": idx, 
                "filename": f, 
                "file_ext": os.path.splitext(f)[1].lower(),
                "CIS ID": get_cis_id(f), 
                "file_path": os.path.join(INPUT_FILES_PATH, f)
            })
        
        df = pd.DataFrame(inventory_data)
        df.to_excel(INVENTORY_FILE_PATH, index=False)
        print(f"--- Inventory created with {len(df)} files. Saved to {INVENTORY_FILE_PATH} ---")
        return df

def process_pipeline(start_index=None, end_index=None):
    
    # 1. Load Inventory (The Master List)
    df_inventory = get_or_create_inventory()
    
    if df_inventory.empty:
        print("No files to process. Exiting.")
        return

    # 2. Determine Processing Range
    # If no index provided, run everything.
    if start_index is None: start_index = 0
    if end_index is None: end_index = len(df_inventory) - 1

    # Bounds check
    if start_index < 0: start_index = 0
    if end_index >= len(df_inventory): end_index = len(df_inventory) - 1

    print(f"--- Pipeline Started: Processing Index {start_index} to {end_index} (High-Res Mode) ---")
    if not RUN_CONTRACT_SCRIPT:
        print(">>> MODE: CONVERSION ONLY (Skipping Contract Scripts) <<<")

    index = start_index

    # 3. Main Loop
    while index <= end_index:
        try:
            # Access row by Index matching the 'inventory.xlsx' structure
            row = df_inventory.iloc[index]
            
            # Validation: Ensure we are processing the correct file for the index
            # (Just in case the excel was sorted or modified, we trust the row data)
            f_name = row['filename']
            f_path = row['file_path']
            f_ext = row['file_ext']
            cis_id = row['CIS ID']
            
            # Construct output name: filename_filetype.txt
            clean_ext = f_ext.replace('.', '')
            final_txt_name = f"{f_name}_{clean_ext}.txt"
            final_txt_path = os.path.join(TXT_OUTPUT_PATH, final_txt_name)

            print(f"\n[{index}] Processing: {f_name} (CIS: {cis_id})")

            # --- A. SKIP CHECK ---
            # If the .txt file already exists, we skip extraction and jump to Step 5
            if os.path.exists(final_txt_path):
                print(f"   -> Output found. Skipping extraction.")
                with open(final_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt_content = f.read()
                goto_step_5(f_name, clean_ext, cis_id, txt_content, f_path, final_txt_path)
                index += 1
                continue

            # --- B. FILE PROCESSING (Based on Type) ---
            
            # VALIDATION: Check if file physically exists
            if not os.path.exists(f_path):
                print(f"   -> ERROR: File not found on disk: {f_path}")
                log_process_status(f"MISSING FILE: {f_name}")
                index += 1
                continue

            # TYPE 1: SPREADSHEETS (Excel/CSV)
            if f_ext in ['.xlsx', '.xls', '.csv']:
                print("   -> Type: Spreadsheet")
                content = ""
                if f_ext == '.csv':
                    content = pd.read_csv(f_path).to_string()
                else:
                    xls = pd.ExcelFile(f_path)
                    for sheet in xls.sheet_names:
                        content += f"##-- SHEET: {sheet} --##\n{pd.read_excel(xls, sheet_name=sheet).to_string()}\n"
                
                with open(final_txt_path, 'w', encoding='utf-8') as f: f.write(content)
                goto_step_5(f_name, clean_ext, cis_id, content, f_path, final_txt_path)

            # TYPE 2: PLAIN TEXT
            elif f_ext == '.txt':
                print("   -> Type: Text File")
                shutil.copy(f_path, final_txt_path)
                with open(final_txt_path, 'r', encoding='utf-8') as f: content = f.read()
                goto_step_5(f_name, clean_ext, cis_id, content, f_path, final_txt_path)

            # TYPE 3: COMPLEX DOCUMENTS (PDF, DOCX, IMAGES)
            elif f_ext in ['.pdf', '.docx', '.tiff', '.tif', '.jpg', '.png', '.jpeg']:
                print("   -> Type: Complex Document (High Res)")
                temp_pdf_path = f_path
                is_docx = (f_ext == '.docx')
                
                # Conversion: DOCX -> PDF
                if is_docx:
                    print("      -> Converting DOCX to PDF...")
                    temp_pdf_path = os.path.join(TXT_OUTPUT_PATH, f"temp_{cis_id}.pdf")
                    docx_to_pdf_convert(f_path, temp_pdf_path)

                images = []
                
                # Rasterize PDF to Images at High DPI (500)
                if f_ext in ['.pdf', '.docx']:
                    try: 
                        print("      -> Rasterizing PDF at 500 DPI...")
                        images = convert_from_path(temp_pdf_path, dpi=PDF_CONVERSION_DPI)
                    except: pass
                elif f_ext in ['.tiff', '.tif']:
                    img = Image.open(f_path)
                    for i in range(getattr(img, 'n_frames', 1)):
                        img.seek(i)
                        images.append(img.copy())
                else:
                    images = [Image.open(f_path)]

                total_extracted_text = ""

                # Loop through every page image
                for i, pil_image in enumerate(images):
                    page_num = i + 1
                    print(f"      -> Page {page_num}/{len(images)}")
                    
                    # Save temporary image for Complexity Check
                    temp_img_path = f"temp_page_{page_num}.png"
                    pil_image.save(temp_img_path)
                    
                    # 1. Check Complexity (Table/Image?)
                    is_complex, token_cost = check_page_complexity(temp_img_path)
                    page_text = ""
                    current_tokens = token_cost

                    if is_complex:
                        print("         -> Complex (LLM High-Res).")
                        # Step A: Deskew
                        deskewed = deskew_image(pil_image)
                        # Step B: Smart Upscale & Sharpen
                        final_img = enhance_and_upscale(deskewed)
                        
                        # Step C: Send to GPT-4o
                        txt_llm, t_ocr = llm_convert_to_text(final_img)
                        page_text = txt_llm
                        current_tokens += t_ocr
                        
                        # Log cost immediately
                        log_token_usage_excel(f_name, current_tokens, 0)
                    else:
                        print("         -> Simple (Digital/OCR).")
                        digital_text = ""
                        has_digital = False
                        
                        # Check if digital text is available (Fast & Free)
                        target_pdf = temp_pdf_path if (is_docx or f_ext == '.pdf') else None
                        
                        if target_pdf and os.path.exists(target_pdf):
                            try:
                                with fitz.open(target_pdf) as doc:
                                    if i < len(doc):
                                        digital_text = doc[i].get_text()
                                        if len(digital_text.strip()) > 15: has_digital = True
                            except: pass

                        if has_digital:
                            page_text = digital_text
                        else:
                            # Fallback to Tesseract OCR (on the high-res image)
                            page_text = pytesseract.image_to_string(pil_image)

                    # Append extracted text
                    formatted_page = f"\n##-- PAGE: {page_num} --##\n{page_text}\n"
                    total_extracted_text += formatted_page
                    
                    # Write to file immediately (Safe against crashes)
                    with open(final_txt_path, 'a', encoding='utf-8') as f: f.write(formatted_page)
                    
                    # Cleanup temp image
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)

                # Cleanup temp PDF (if DOCX)
                if is_docx and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                
                # Proceed to Finalize
                goto_step_5(f_name, clean_ext, cis_id, total_extracted_text, f_path, final_txt_path)

            else:
                print(f"   [Skipped] Unknown file type: {f_ext}")
                log_process_status(f"SKIPPED: {f_name}")

            index += 1

        except Exception as e:
            # --- ERROR HANDLING ---
            # If an error occurs, log it and move to the next file. DO NOT HALT.
            print(f"!!! ERROR on Index {index}: {e}")
            log_process_status(f"ERROR: Index {index} - {str(e)}")
            index += 1
            time.sleep(1) # Brief pause to stabilize
            continue

def goto_step_5(filename, filetype, cis_id, text_content, original_path, txt_path):
    """
    Step 5: Finalize.
    If RUN_CONTRACT_SCRIPT is False, it stops here.
    If True, it identifies the contract type and runs the corresponding external script.
    """
    if not RUN_CONTRACT_SCRIPT:
        print(f"   -> Conversion Complete. Saved to: {os.path.basename(txt_path)}")
        log_process_status(f"CONVERTED ONLY: {filename}")
        return
    
    print("   -> Step 5: Contract Analysis")
    # Identify Type using LLM
    ctype = determine_contract_type(text_content)
    print(f"      -> Identified Type: {ctype}")
    
    # Locate Script
    target_script = f"{ctype}.py"
    script_full_path = os.path.join(CONTRACT_SCRIPTS_PATH, target_script)
    
    # Fallback to others.py if specific script missing
    if not os.path.exists(script_full_path):
        script_full_path = os.path.join(CONTRACT_SCRIPTS_PATH, "others.py")
        ctype_arg = ctype 
    else:
        ctype_arg = ctype

    # Execute Script
    if os.path.exists(script_full_path):
        try:
            subprocess.run([
                "python", script_full_path,
                "--filename", filename, "--filetype", filetype, "--cis_id", str(cis_id),
                "--contract_type", ctype_arg, "--file_path", original_path, "--txt_path", txt_path
            ], check=True)
            log_process_status(f"SUCCESS: {filename} processed as {ctype}")
        except subprocess.CalledProcessError as e:
            print(f"      [Error] Subprocess Failed: {e}")
    else:
        print("      [Error] 'others.py' is missing.")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # ARGUMENT PARSING
    # This allows you to run: "python script.py --start 0 --end 100"
    parser = argparse.ArgumentParser(description="Medical Contract OCR Pipeline")
    parser.add_argument("--start", type=int, default=None, help="Start Index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End Index (inclusive)")
    
    args = parser.parse_args()
    
    # Run the pipeline with provided arguments
    process_pipeline(start_index=args.start, end_index=args.end)
