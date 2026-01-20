import os
import re
import cv2
import shutil
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from docx import Document
from scipy.ndimage import rotate
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo

# --- CONFIGURATION ---
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

INPUT_FOLDER = './input_docs'
TEXT_OUTPUT_FOLDER = './converted_text'
FINAL_EXCEL_PATH = 'Split_Bill_Extraction_Report.xlsx'
EXT_ID = "SIV0592"

# --- DESKEWING LOGIC ---
def deskew_page(page_images_path, temp_folder):
    deskewed_folder = os.path.join(temp_folder, "deskewed_images")
    os.makedirs(deskewed_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(page_images_path) if f.lower().endswith('.png')])
    
    for filename in image_files:
        input_path = os.path.join(page_images_path, filename)
        image = cv2.imread(input_path)
        if image is not None:
            h, w = image.shape[:2]
            scale = 800 / w
            small_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            scores = []
            angles = np.arange(-5, 5.1, 0.1)
            for angle in angles:
                rotated = rotate(thresh, angle, reshape=False, order=0)
                projection = np.sum(rotated, axis=1)
                scores.append(np.var(projection))
            best_angle = angles[np.argmax(scores)]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
            deskewed_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            cv2.imwrite(os.path.join(deskewed_folder, filename), deskewed_image)
    return deskewed_folder

# --- PHASE 1: CONVERSION ---
def convert_to_text():
    print("PHASE 1: Converting files to .txt...")
    if not os.path.exists(TEXT_OUTPUT_FOLDER): os.makedirs(TEXT_OUTPUT_FOLDER)

    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            file_path = os.path.join(root, file)
            ext = file.lower().split('.')[-1]
            output_txt_path = os.path.join(TEXT_OUTPUT_FOLDER, f"{file}.txt")
            if os.path.exists(output_txt_path): continue
            
            text_content = ""
            try:
                if ext == 'pdf':
                    images = convert_from_path(file_path)
                    temp_img_dir = "temp_pages"
                    os.makedirs(temp_img_dir, exist_ok=True)
                    for i, img in enumerate(images): img.save(os.path.join(temp_img_dir, f"p_{i}.png"), "PNG")
                    deskewed_dir = deskew_page(temp_img_dir, "temp_work")
                    for img_f in sorted(os.listdir(deskewed_dir)):
                        text_content += pytesseract.image_to_string(Image.open(os.path.join(deskewed_dir, img_f))) + "\n"
                    shutil.rmtree(temp_img_dir)
                    if os.path.exists("temp_work"): shutil.rmtree("temp_work")
                elif ext == 'docx':
                    text_content = "\n".join([p.text for p in Document(file_path).paragraphs])
                elif ext in ['xlsx', 'xls']:
                    text_content = pd.read_excel(file_path).to_string()
                elif ext == 'txt':
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: text_content = f.read()

                with open(output_txt_path, 'w', encoding='utf-8') as f: f.write(text_content)
                print(f"Successfully converted: {file}")
            except Exception as e: print(f"Error converting {file}: {e}")

# --- PHASE 2: EXTRACTION & EXCEL FORMATTING ---
def format_excel_table(file_path):
    wb = load_workbook(file_path)
    ws = wb.active
    if ws.max_row > 1:
        tab = Table(displayName="ExtractionTable", ref=f"A1:{chr(64 + ws.max_column)}{ws.max_row}")
        tab.tableStyleInfo = TableStyleInfo(name="TableStyleMedium9", showRowStripes=True)
        ws.add_table(tab)
    for col in ws.columns:
        max_length = 0
        for cell in col:
            if cell.value: max_length = max(max_length, len(str(cell.value)))
        ws.column_dimensions[col[0].column_letter].width = min(max_length + 2, 70)
    wb.save(file_path)

def run_extraction():
    print("PHASE 2: Running Regex Extraction...")
    results = []
    # REGEX: Captures from the start of the string or last full stop, 
    # through the keyword, up to the next full stop.
    regex_pattern = r"(?:^|(?<=[.!?]))\s*([^.!?]*?FOR THE SAME ILLNESS OR INJURY[^.!?]*?[.!?])"

    for txt_file in os.listdir(TEXT_OUTPUT_FOLDER):
        cis_id = (re.search(r'(\d{6})', txt_file) or [None, "N/A"])[1]
        file_ext = txt_file.split('.')[-2]
        
        with open(os.path.join(TEXT_OUTPUT_FOLDER, txt_file), 'r', encoding='utf-8') as f:
            content = " ".join(f.read().split()) # Clean whitespace/newlines
            matches = re.findall(regex_pattern, content, re.IGNORECASE)
            
            if matches:
                for match in matches:
                    results.append({
                        'File_name': txt_file.replace('.txt', ''),
                        'File_ext': file_ext,
                        'CIS_ID': cis_id,
                        'Lang_Ind': 1,
                        'Lang': match.strip(),
                        'EXT_ID': EXT_ID,
                        'EXT_Date_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
            else:
                results.append({
                    'File_name': txt_file.replace('.txt', ''), 'File_ext': file_ext,
                    'CIS_ID': cis_id, 'Lang_Ind': 0, 'Lang': "N/A",
                    'EXT_ID': EXT_ID, 'EXT_Date_Time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    pd.DataFrame(results).to_excel(FINAL_EXCEL_PATH, index=False)
    format_excel_table(FINAL_EXCEL_PATH)
    print(f"Process complete. Output saved to: {FINAL_EXCEL_PATH}")

def main():
    convert_to_text() # Phase 1
    run_extraction()  # Phase 2

if __name__ == "__main__":
    main()
