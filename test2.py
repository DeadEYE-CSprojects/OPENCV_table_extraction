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


import pytesseract
import shutil
from PIL import Image, ImageDraw

def check_pytesseract_ready():
    print("--- Checking Pytesseract Configuration ---")

    # 1. Check for the Tesseract Binary (System Level)
    # This checks if 'apt-get install tesseract-ocr' worked
    tesseract_path = shutil.which("tesseract")
    
    if tesseract_path:
        print(f"✅ [System] Tesseract binary found at: {tesseract_path}")
    else:
        print("❌ [System] Tesseract binary NOT found.")
        print("   -> Solution: Run '%sh apt-get update && apt-get install -y tesseract-ocr' in a separate cell.")
        return False

    # 2. Check Python Binding & Version
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ [Python] Pytesseract connected. Engine Version: {version}")
    except Exception as e:
        print(f"❌ [Python] Error communicating with Tesseract: {e}")
        return False

    # 3. Test Actual OCR on a generated image
    print("\n--- Running Functional OCR Test ---")
    try:
        # Create a simple white image with black text "OK"
        img = Image.new('RGB', (60, 30), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "OK", fill=(0, 0, 0))
        
        # Perform OCR
        # --psm 7 treats the image as a single text line (good for small test images)
        text = pytesseract.image_to_string(img, config='--psm 7')
        
        print(f"✅ [OCR Test] Engine output: '{text.strip()}'")
        return True

    except Exception as e:
        print(f"❌ [OCR Test] Failed to process image: {e}")
        return False

# Execute the check
if check_pytesseract_ready():
    print("\n🎉 SUCCESS: System is fully ready for OCR.")
else:
    print("\n🚫 FAILURE: System is not ready.")
