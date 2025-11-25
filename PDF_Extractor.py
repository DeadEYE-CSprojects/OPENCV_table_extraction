import cv2
import os
import glob
import pandas as pd
import base64
import httpx
from openai import OpenAI
import json
import re

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# --- PATHS ---
BASE_OUTPUT_FOLDER = 'final_data' 
FINAL_EXCEL_PATH = os.path.join(BASE_OUTPUT_FOLDER, 'Batch_Extraction_Report.xlsx')

# --- LLM SETUP ---
API_KEY = "YOUR_API_KEY"
BASE_URL = "https://gateway.ai.humana.com/openai" 
CERT_PATH = '/Volumes/data_and_analytics_main_qa_000/capi_gqd/testnlp/test/_.ai.humana.pem'

# =============================================================================
# 2. SETUP CLIENT
# =============================================================================
try:
    http_client = httpx.Client(verify=CERT_PATH)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)
    print("✅ LLM Client Connected.")
except Exception as e:
    print(f"❌ Error connecting client: {e}")
    exit()

# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def encode_image_from_path(image_path):
    """Reads an image file from disk and converts to Base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_region_1_data(image_path):
    """
    Analyzes Region 1 for Blocks 31, 32 (a/b), and 33 (a/b).
    """
    if not os.path.exists(image_path):
        return {}

    base64_img = encode_image_from_path(image_path)
    
    prompt = """
    Analyze this section of the CMS-1500 form (Blocks 31, 32, and 33).
    Extract the following data into a specific JSON structure.
    
    1. **Block 31 (Signature)**: Look for "Signed" status (Yes/No), the name/text found, and the Date.
    2. **Block 32 (Service Facility)**: Extract the Name/Address, 32a (NPI), and 32b (Other ID).
    3. **Block 33 (Billing Provider)**: Extract the Name/Address/Phone, 33a (NPI), and 33b (Other ID).

    Return ONLY a valid JSON object with these keys. If a field is empty, use "Empty".
    
    {
      "Block_31_Is_Signed": "Yes/No",
      "Block_31_Signature_Text": "Text found",
      "Block_31_Date": "Date or Empty",
      
      "Block_32_Facility_Name_Address": "Full text found",
      "Block_32a_NPI": "Number or Empty",
      "Block_32b_Other_ID": "Number or Empty",
      
      "Block_33_Billing_Name_Address_Phone": "Full text found",
      "Block_33a_NPI": "Number or Empty",
      "Block_33b_Other_ID": "Number or Empty"
    }
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a medical claims data extractor."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"}}
                ]}
            ],
            temperature=0,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"   ⚠️ Region 1 LLM Error: {e}")
        return {"Block_31_Is_Signed": "Error"}

def extract_region_2_data(image_path):
    """
    Analyzes Region 2 (Block 24) specifically for columns I and J.
    """
    if not os.path.exists(image_path):
        return {}

    base64_img = encode_image_from_path(image_path)
    
    prompt = """
    Focus on Block 24 (Service Lines). Look for the columns labeled "24 I" (ID Qualifier) and "24 J" (Rendering Provider ID).
    
    Extract the data from the FIRST service line available.
    
    Return a JSON object with these exact keys:
    {
      "Block_24_I_Qualifier": "Value in column 24 I (e.g., ZZ, 1G) or Empty",
      "Block_24_J_Rendering_Provider_ID": "Value in column 24 J (NPI number) or Empty"
    }
    Return ONLY JSON.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a precise data extractor."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}", "detail": "high"}}
                ]}
            ],
            temperature=0,
            max_tokens=300
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"   ⚠️ Region 2 LLM Error: {e}")
        return {"Block_24_J_Rendering_Provider_ID": "Error"}

# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================

def main():
    print("\n=== STARTING FOLDER-BASED EXTRACTION ===")
    
    subfolders = [f.path for f in os.scandir(BASE_OUTPUT_FOLDER) if f.is_dir()]
    all_rows = []
    
    print(f"Found {len(subfolders)} file folders to process.")

    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"Processing: {folder_name}...")
        
        r1_path = os.path.join(folder_path, 'Region_1.png')
        r2_path = os.path.join(folder_path, 'Region_2.png')
        
        current_row = {'Filename': folder_name}
        
        # --- PROCESS REGION 1 (Blocks 31, 32, 33) ---
        if os.path.exists(r1_path):
            r1_data = extract_region_1_data(r1_path)
            current_row.update(r1_data)
        else:
            print(f"   ⚠️ Missing Region_1.png in {folder_name}")

        # --- PROCESS REGION 2 (Block 24 I/J) ---
        if os.path.exists(r2_path):
            r2_data = extract_region_2_data(r2_path)
            current_row.update(r2_data)
        else:
            print(f"   ⚠️ Missing Region_2.png in {folder_name}")
            
        all_rows.append(current_row)

    # --- SAVE TO EXCEL ---
    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # Define the exact column order you want
        cols = [
            'Filename', 
            'Block_31_Is_Signed', 'Block_31_Signature_Text', 'Block_31_Date',
            'Block_32_Facility_Name_Address', 'Block_32a_NPI', 'Block_32b_Other_ID',
            'Block_33_Billing_Name_Address_Phone', 'Block_33a_NPI', 'Block_33b_Other_ID',
            'Block_24_I_Qualifier', 'Block_24_J_Rendering_Provider_ID'
        ]
        
        # Reindex to enforce order and create missing columns with "N/A"
        df = df.reindex(columns=cols, fill_value="N/A")
        
        df.to_excel(FINAL_EXCEL_PATH, index=False)
        print(f"\n✅ SUCCESS! Report saved to: {FINAL_EXCEL_PATH}")
        print(df.head())
    else:
        print("\n❌ No data found.")

if __name__ == "__main__":
    main()
