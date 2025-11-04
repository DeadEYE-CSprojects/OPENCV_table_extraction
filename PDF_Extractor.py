import fitz  # PyMuPDF
import docx
import openpyxl
import base64
import io
import os
import json
import pandas as pd
import numpy as np
import time
from openai import OpenAI

# --- 1. OpenAI Client Initialization ---
# 
# ❗ IMPORTANT: Initialize your client here with your API key
# client = OpenAI(api_key="YOUR_API_KEY_HERE")
#
# --- (End of Client Initialization) ---


# --- 2. (Part 1) Pre-processing Function ---

def preprocess_document(file_path: str) -> list:
    """
    Takes a file path (pdf, docx, xlsx) and converts it into
    a standardized list of text and image content blocks.
    """
    print(f"--- Starting preprocessing for: {file_path} ---")
    content_blocks = []
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.pdf':
            doc = fitz.open(file_path)
            print(f"Processing {len(doc)} PDF pages...")
            for page_num, page in enumerate(doc):
                # 1. Extract Text
                text = page.get_text()
                if text.strip():
                    content_blocks.append({
                        "type": "text",
                        "content": f"--- PDF Page {page_num + 1} Text ---\n{text}"
                    })
                
                # 2. Extract Images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    content_blocks.append({
                        "type": "image",
                        "content": image_b64
                    })
            doc.close()

        elif file_ext == '.docx':
            doc = docx.Document(file_path)
            print("Processing .docx file...")
            # 1. Extract Text
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            if full_text:
                content_blocks.append({
                    "type": "text",
                    "content": "\n".join(full_text)
                })
            # 2. Extract Images
            for i, shape in enumerate(doc.inline_shapes):
                rId = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
                image_part = doc.part.related_parts[rId]
                image_bytes = image_part.blob
                image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                content_blocks.append({
                    "type": "image",
                    "content": image_b64
                })

        elif file_ext == '.xlsx':
            wb = openpyxl.load_workbook(file_path)
            print(f"Processing {len(wb.sheetnames)} Excel sheets...")
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                # 1. Extract Text (as CSV-like string)
                sheet_text = []
                for row in ws.iter_rows():
                    row_text = [str(cell.value) if cell.value is not None else "" for cell in row]
                    sheet_text.append(",".join(row_text))
                if sheet_text:
                    content_blocks.append({
                        "type": "text",
                        "content": f"--- Excel Sheet '{sheet_name}' ---\n" + "\n".join(sheet_text)
                    })
                # 2. Extract Images
                if hasattr(ws, '_images') and ws._images:
                    for i, img in enumerate(ws._images):
                        image_bytes = img.data()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        content_blocks.append({
                            "type": "image",
                            "content": image_b64
                        })
            wb.close()
        else:
            # Fallback for .txt or other formats
            with open(file_path, 'r', encoding='utf-8') as f:
                content_blocks.append({"type": "text", "content": f.read()})
                
        print(f"--- Preprocessing complete. {len(content_blocks)} blocks created. ---")
        return content_blocks
    except Exception as e:
        print(f"ERROR during preprocessing: {e}")
        return []

# --- 3. (Part 2) General Extraction Logic ---

def load_all_examples(excel_path='keywords_patters.xlsx') -> dict:
    """
    UPDATED: Loads examples from Excel.
    Assumes first column is 'Field'.
    Combines all other columns with " or ".
    """
    print(f"Loading examples from {excel_path}...")
    try:
        df = pd.read_excel(excel_path) 
        
        field_col_name = df.columns[0]
        example_col_names = df.columns[1:]
        
        print(f"Reading fields from column: '{field_col_name}'")
        print(f"Reading examples from {len(example_col_names)} columns.")
        
        examples = {}
        
        for index, row in df.iterrows():
            field_name = row[field_col_name]
            if not field_name:
                continue
                
            example_list = []
            for col in example_col_names:
                example = row[col]
                if pd.notna(example) and str(example).strip():
                    example_list.append(str(example).strip())
            
            if example_list:
                # Combine with " or " as you requested
                combined_example_string = " or ".join(example_list)
                examples[field_name] = combined_example_string
            else:
                examples[field_name] = "None provided."
                
        print(f"Successfully loaded and combined examples for {len(examples)} fields.")
        return examples
    
    except FileNotFoundError:
        print(f"ERROR: Example file not found at {excel_path}")
        return {}
    except Exception as e:
        print(f"ERROR loading examples: {e}")
        return {}

def build_multimodal_message(instruction_prompt: str, content_blocks: list) -> list:
    """
    Helper function to combine the instruction prompt with all
    the text and image blocks for the API call.
    """
    message_content = []
    message_content.append({"type": "text", "text": instruction_prompt})
    
    for block in content_blocks:
        if block['type'] == 'text':
            message_content.append({"type": "text", "text": block['content']})
        elif block['type'] == 'image':
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{block['content']}",
                    "detail": "high"
                }
            })
    return message_content

def extract_data_from_document(content_blocks: list, examples: dict) -> dict:
    """
    Runs the general-purpose, multi-pass LLM extraction.
    """
    combined_results = {}

    # --- PASS 1: Entity, Data, and Classification ---
    print("--- Starting Pass 1: Entity/Data Extraction ---")
    try:
        prompt1 = create_entity_prompt(examples)
        message1_content = build_multimodal_message(prompt1, content_blocks)
        
        # === ❗ UNCOMMENT TO RUN ❗ ===
        # response1 = client.chat.completions.create(
        #     model="gpt-4o",  # Must be a multimodal model
        #     response_format={"type": "json_object"},
        #     messages=[{"role": "user", "content": message1_content}],
        #     max_tokens=4096
        # )
        # entity_data = json.loads(response1.choices[0].message.content)
        # ============================

        # --- Simulated data for testing (DELETE THIS WHEN RUNNING) ---
        entity_data = {
            "CIS_TYPE": "AEC",
            "CIS_TYPE_DESCRIPTION": "Ambulatory emergency services",
            "ATTACHMENT_ID": "Attach-123",
            "PROVIDER_NAME": "General Hospital",
            "NPI": "1234567890",
            "TAX_ID": "XX-1234567",
            "REIMBURSEMENT_AMT": "5000"
        }
        # --- End of simulated data ---

        print("Pass 1 successful.")
        combined_results.update(entity_data)

    except Exception as e:
        print(f"ERROR in Pass 1 (Entities): {e}")
        combined_results["NLP_ERROR_COMMENTS"] = f"Pass 1 Failed: {e}"


    # --- PASS 2: Indicators and Language Clauses ---
    print("--- Starting Pass 2: Indicator/Language Extraction ---")
    try:
        prompt2 = create_language_prompt(examples)
        message2_content = build_multimodal_message(prompt2, content_blocks)
        
        # === ❗ UNCOMMENT TO RUN ❗ ===
        # response2 = client.chat.completions.create(
        #     model="gpt-4o",  # Must be a multimodal model
        #     response_format={"type": "json_object"},
        #     messages=[{"role": "user", "content": message2_content}],
        #     max_tokens=4096
        # )
        # language_data = json.loads(response2.choices[0].message.content)
        # ============================

        # --- Simulated data for testing (DELETE THIS WHEN RUNNING) ---
        language_data = {
          "INDICATORS": {
            "CPT_IND": 1,
            "EXCLUSION_IND": 1,
            "DRG_CD_IND": 0
          },
          "LANGUAGE_CLAUSES": {
            "READMISSION_LANG": {
              "IND": 1,
              "TIMEFRAME": "30 days",
              "LANG": "Re-hospitalization within 30 days is not covered."
            },
            "MODIFICATION_LANG": {
              "IND": 1,
              "NOTICE_PERIOD": "90 days",
              "LANG": "This Agreement may be modified by Plan with 90 days written notice."
            }
          }
        }
        # --- End of simulated data ---

        flattened_lang_data = flatten_language_json(language_data)
        combined_results.update(flattened_lang_data)
        print("Pass 2 successful.")
        
    except Exception as e:
        print(f"ERROR in Pass 2 (Language): {e}")
        if combined_results.get("NLP_ERROR_COMMENTS"):
            combined_results["NLP_ERROR_COMMENTS"] += f" | Pass 2 Failed: {e}"
        else:
            combined_results["NLP_ERROR_COMMENTS"] = f"Pass 2 Failed: {e}"


    return combined_results

# --- 4. Prompt and Helper Functions ---

def create_entity_prompt(examples: dict) -> str:
    """
    Creates the prompt for Pass 1 (Entities, Data, Classification).
    Uses underscored field names and "or"-joined examples.
    """
    def get_ex_str(key):
        return examples.get(key, "None provided.")

    return f"""
    You are an expert data extractor for US healthcare contracts.
    Analyze the following document pages (text and images)
    and extract the following information.
    Return ONLY a valid JSON object with the specified UNDERSCORED keys.
    If a value is not found, return null.

    ---
    CLASSIFICATION (Critical)
    
    1.  **CIS_TYPE**: Identify the document type. It must be one of:
        * AEC
        * CAH
        * Anesthesia
        * ASC
        * Audiology & hearing
        * Cardiology heart and vascular
    
    2.  **CIS_TYPE_DESCRIPTION**: If CIS_TYPE is 'AEC', set this to 'Ambulatory emergency services'.
        If 'CAH', set to 'Critical Access Hospital'.
        If 'ASC', set to 'Ambulatory surgery center'.
        Otherwise, set to null.

    ---
    DATA EXTRACTION

    Extract the following fields. Use the examples as hints for the types of
    patterns to look for.
    
    * ATTACHMENT_ID: (Any ID found inside the document text)
    * PROVIDER_NAME: (Examples: {get_ex_str('PROVIDER_NAME')})
    * NPI: (Examples: {get_ex_str('NPI')})
    * TAX_ID: (Examples: {get_ex_str('TAX_ID')})
    * PROVZIPCODE: (Examples: {get_ex_str('PROVZIPCODE')})
    * TAXONOMYCODE: (Examples: {get_ex_str('TAXONOMYCODE')})
    * EFFECTIVE_FROM_DATE: (Format YYYY-MM-DD. Examples: {get_ex_str('EFFECTIVE_FROM_DATE')})
    * EFFECTIVE_TO_DATE: (Format YYYY-MM-DD. Examples: {get_ex_str('EFFECTIVE_TO_DATE')})
    * PLACEOFSERV: (Examples: {get_ex_str('PLACEOFSERV')})
    * LOB_IND: (Examples: {get_ex_str('LOB_IND')})
    * SERVICE_TYPE: (Examples: {get_ex_str('SERVICE_TYPE')})
    * SERVICE_DESC: (Examples: {get_ex_str('SERVICE_DESC')})
    * SERVICES: (Examples: {get_ex_str('SERVICES')})
    * AGE_GROUP: (e.g., adult, child, 0-18. Examples: {get_ex_str('AGE_GROUP')})
    * CODES: (Any CPT, service, revenue codes. Examples: {get_ex_str('CODES')})
    * GROUPER: (Examples: {get_ex_str('GROUPER')})
    * CASE_RATE: (Examples: {get_ex_str('CASE_RATE')})
    * DISCHARGESTATUSCODE: (Examples: {get_ex_str('DISCHARGESTATUSCODE')})
    * ALOSGLOS: (Examples: {get_ex_str('ALOSGLOS')})
    * TRANSFER_RATE: (Examples: {get_ex_str('TRANSFER_RATE')})
    * APPLIEDTRANSFERCASE: (Return 'yes' or 'no'. Examples: {get_ex_str('APPLIEDTRANSFERCASE')})
    * REIMBURSEMENT_AMT: (Examples: {get_ex_str('REIMBURSEMENT_AMT')})
    * REIMBURSEMENT_RATE: (Examples: {get_ex_str('REIMBURSEMENT_RATE')})
    * REIMBURSEMENT_METHODOLOGY: (Examples: {get_ex_str('REIMBURSEMENT_METHODOLOGY')})
    * MULTIPLERMMETHODS: (Examples: {get_ex_str('MULTIPLERMMETHODS')})
    * METHOD_OF_PAYMENT: (Examples: {get_ex_str('METHOD_OF_PAYMENT')})
    * HEALTH_BENEFIT_PLANS: (Find tables, look for marked items)
    * PROVIDER_SPECIALITY: (Examples: {get_ex_str('PROVIDER_SPECIALITY')})
    * OTHER_FLAT_FEE: (Examples: {get_ex_str('OTHER_FLAT_FEE')})
    * SURG_FLAT_FEE: (Examples: {get_ex_str('SURG_FLAT_FEE')})
    * AND_OR_OPERATOR: (Examples: {get_ex_str('AND_OR_OPERATOR')})
    * OPERATOR_CODE_TYPE: (Examples: {get_ex_str('OPERATOR_CODE_TYPE')})
    * ADDITIONAL_NOTES: (Any other key details or notes for the user)
    
    JSON OUTPUT:
    """

def create_language_prompt(examples: dict) -> str:
    """
    Creates the prompt for Pass 2 (Indicators and Language Clauses).
    Uses underscored field names and "or"-joined examples.
    """
    
    def get_ex_str(key):
        return examples.get(key, "None provided.")

    return f"""
    You are an expert legal analyst for US healthcare contracts.
    Your task is to read the document pages (text and images) 
    and identify all specified indicators and language clauses.
    
    You MUST return ONLY a valid JSON object with the exact nested structure specified below.

    ---
    EXAMPLES OF LANGUAGE TO LOOK FOR (Find similar semantic meaning):
    
    * READMISSION_LANG Examples:
        {get_ex_str('READMISSION_LANG')}
    * TERMINATION_LANG Examples:
        {get_ex_str('TERMINATION_LANG')}
    * MODIFICATION_LANG Examples:
        {get_ex_str('MODIFICATION_LANG')}
    * LABS_LANG Examples:
        {get_ex_str('LABS_LANG')}
    * NEW_PROD_LANG Examples:
        {get_ex_str('NEW_PROD_LANG')}
    * LER_LANG Examples:
        {get_ex_str('LER_LANG')}
    * (and so on for all other language types...)
    ---
    
    NOW, analyze the document and provide the JSON output.
    
    1.  **INDICATORS**: Set to 1 if the concept is present anywhere, 0 otherwise.
    2.  **LANGUAGE_CLAUSES**: For each clause:
        * Set "IND" to 1 if found, 0 otherwise.
        * Set "LANG" to the full text of the clause you found.
        * Extract the related data (e.g., TIMEFRAME, CODES, NOTICE_PERIOD).

    JSON OUTPUT STRUCTURE:
    {{
      "INDICATORS": {{
        "REVENUE_CD_IND": 0,
        "DRG_CD_IND": 0,
        "CPT_IND": 0,
        "HCPCS_IND": 0,
        "ICD_CD_IND": 0,
        "DIAGNOSIS_CD_IND": 0,
        "MODIFIER_CD_IND": 0,
        "GROUPER_IND": 0,
        "APC_IND": 0,
        "EXCLUSION_IND": 0,
        "MSR_IND": 0,
        "BILETRAL_PROCEDURE_IND": 0,
        "EXCLUDE_FROM_TRANSFER_IND": 0,
        "EXCLUDE_FROM_STOPLOSS_IND": 0,
        "ISTHRESHOLD": 0,
        "ISCAPAMOUNT": 0
      }},
      "LANGUAGE_CLAUSES": {{
        "READMISSION_LANG": {{"IND": 0, "TIMEFRAME": null, "LANG": null}},
        "LABS_LANG": {{"IND": 0, "CODES": null, "LANG": null}},
        "MODIFICATION_LANG": {{"IND": 0, "NOTICE_PERIOD": null, "LANG": null}},
        "NEW_PROD_LANG": {{"IND": 0, "NOTICE_PERIOD": null, "LANG": null}},
        "INTRA_FACIILITY_TRANSFER_LANG": {{"IND": 0, "TIMEFRAME": null, "LANG": null}},
        "POST_DISCHG_TESTING_LANG": {{"IND": 0, "TIMEFRAME": null, "LANG": null}},
        "CASE_RATE_PER_ADMIT_LANG": {{"IND": 0, "AMT": null, "LANG": null}},
        "LER_LANG": {{"IND": 0, "TIMEFRAME": null, "LANG": null}},
        "TERMINATION_LANG": {{"IND": 0, "NOTICE_PERIOD": null, "LANG": null}},
        "MANUAL_ADM_LANG": {{"IND": 0, "DATE": null, "LANG": null}},
        "MEDICARE_INPATIENT_PPS_LANG": {{"IND": 0, "LANG": null}}
      }}
    }}
    """

def flatten_language_json(nested_json: dict) -> dict:
    """
    Converts the nested JSON from Pass 2 into a flat dictionary
    with the final underscored field names.
    """
    flat_data = {}
    
    if "INDICATORS" in nested_json:
        flat_data.update(nested_json["INDICATORS"])
            
    if "LANGUAGE_CLAUSES" in nested_json:
        for lang_key, lang_data in nested_json.get("LANGUAGE_CLAUSES", {}).items():
            if not lang_data: lang_data = {}
            
            if lang_key == "READMISSION_LANG":
                flat_data["READMISSION_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["READMISSION_LANG_TIMEFRAME"] = lang_data.get("TIMEFRAME")
                flat_data["READMISSION_LANG"] = lang_data.get("LANG")
            
            elif lang_key == "LABS_LANG":
                flat_data["LABS_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["LABS_LANG_CODES"] = lang_data.get("CODES")
                flat_data["LABS_LANG"] = lang_data.get("LANG")
                
            elif lang_key == "MODIFICATION_LANG":
                flat_data["MODIFICATION_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["MODIFICATION_LANG_NOTICE_PERIOD"] = lang_data.get("NOTICE_PERIOD")
                flat_data["MODIFICATION_LANG"] = lang_data.get("LANG")

            elif lang_key == "NEW_PROD_LANG":
                flat_data["NEW_PROD_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["NEW_PROD_LANG_NOTICE_PERIOD"] = lang_data.get("NOTICE_PERIOD")
                flat_data["NEW_PROD_LANG"] = lang_data.get("LANG")

            elif lang_key == "INTRA_FACIILITY_TRANSFER_LANG":
                flat_data["INTRA_FACIILITY_TRANSFER_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["INTRA_FACIILITY_TRANSFER_LANG_TIMEFRAME"] = lang_data.get("TIMEFRAME")
                flat_data["INTRA_FACIILITY_TRANSFER_LANG"] = lang_data.get("LANG")

            elif lang_key == "POST_DISCHG_TESTING_LANG":
                flat_data["POST_DISCHG_TESTING_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["POST_DISCHG_TESTING_LANG_TIMEFRAME"] = lang_data.get("TIMEFRAME")
                flat_data["POST_DISCHG_TESTING_LANG"] = lang_data.get("LANG")

            elif lang_key == "CASE_RATE_PER_ADMIT_LANG":
                flat_data["CASE_RATE_PER_ADMIT_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["CASE_RATE_PER_ADMIT_LANG_AMT"] = lang_data.get("AMT")
                flat_data["CASE_RATE_PER_ADMIT_LANG"] = lang_data.get("LANG")

            elif lang_key == "LER_LANG":
                flat_data["LER_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["LER_LANG_TIMEFRAME"] = lang_data.get("TIMEFRAME")
                flat_data["LER_LANG"] = lang_data.get("LANG")

            elif lang_key == "TERMINATION_LANG":
                flat_data["TERMINATION_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["TERMINATION_LANG_NOTICE_PERIOD"] = lang_data.get("NOTICE_PERIOD")
                flat_data["TERMINATION_LANG"] = lang_data.get("LANG")
                
            elif lang_key == "MANUAL_ADM_LANG":
                flat_data["MANUAL_ADM_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["MANUAL_ADM_LANG_DATE"] = lang_data.get("DATE")
                flat_data["MANUAL_ADM_LANG"] = lang_data.get("LANG")

            elif lang_key == "MEDICARE_INPATIENT_PPS_LANG":
                flat_data["MEDICARE_INPATIENT_PPS_LANG_IND"] = lang_data.get("IND", 0)
                flat_data["MEDICARE_INPATIENT_PPS_LANG"] = lang_data.get("LANG")

    return flat_data

def initialize_all_fields() -> dict:
    """
    Creates a dictionary with all fields (underscored)
    set to None or 0. This ensures all columns exist in the output.
    """
    return {
        "CIS_CTRCT_ID": None, "ATTACHMENT_ID": None, "FILE_NAME": None,
        "FILE_EXTENSION": None, "CIS_TYPE": None, "CIS_TYPE_DESCRIPTION": None,
        "PROVIDER_NAME": None, "NPI": None, "TAX_ID": None, "PROVZIPCODE": None,
        "TAXONOMYCODE": None, "EFFECTIVE_FROM_DATE": None,
        "EFFECTIVE_TO_DATE": None, "PLACEOFSERV": None, "LOB_IND": None,
        "SERVICE_TYPE": None, "SERVICE_DESC": None, "SERVICES": None,
        "AGE_GROUP": None, "CODES": None, "GROUPER": None, "CASE_RATE": None,
        "REVENUE_CD_IND": 0, "DRG_CD_IND": 0, "CPT_IND": 0, "HCPCS_IND": 0,
        "ICD_CD_IND": 0, "DIAGNOSIS_CD_IND": 0, "MODIFIER_CD_IND": 0,
        "GROUPER_IND": 0, "APC_IND": 0, "EXCLUSION_IND": 0, "MSR_IND": 0,
        "BILETRAL_PROCEDURE_IND": 0, "EXCLUDE_FROM_TRANSFER_IND": 0,
        "EXCLUDE_FROM_STOPLOSS_IND": 0, "DISCHARGESTATUSCODE": None,
        "ALOSGLOS": None, "TRANSFER_RATE": None, "APPLIEDTRANSFERCASE": None,
        "ISTHRESHOLD": 0, "ISCAPAMOUNT": 0, "REIMBURSEMENT_AMT": None,
        "REIMBURSEMENT_RATE": None, "REIMBURSEMENT_METHODOLOGY": None,
        "MULTIPLERMMETHODS": None, "METHOD_OF_PAYMENT": None,
        "HEALTH_BENEFIT_PLANS": None, "ADDITIONAL_NOTES": None,
        "PROVIDER_SPECIALITY": None, "OTHER_FLAT_FEE": None, "SURG_FLAT_FEE": None,
        "AND_OR_OPERATOR": None, "OPERATOR_CODE_TYPE": None,
        "READMISSION_LANG_IND": 0, "READMISSION_LANG_TIMEFRAME": None,
        "READMISSION_LANG": None, "LABS_LANG_IND": 0, "LABS_LANG_CODES": None,
        "LABS_LANG": None, "MODIFICATION_LANG_IND": 0,
        "MODIFICATION_LANG_NOTICE_PERIOD": None, "MODIFICATION_LANG": None,
        "NEW_PROD_LANG_IND": 0, "NEW_PROD_LANG_NOTICE_PERIOD": None,
        "NEW_PROD_LANG": None, "INTRA_FACIILITY_TRANSFER_LANG_IND": 0,
        "INTRA_FACIILITY_TRANSFER_LANG_TIMEFRAME": None,
        "INTRA_FACIILITY_TRANSFER_LANG": None,
        "POST_DISCHG_TESTING_LANG_IND": 0,
        "POST_DISCHG_TESTING_LANG_TIMEFRAME": None,
        "POST_DISCHG_TESTING_LANG": None, "CASE_RATE_PER_ADMIT_LANG_IND": 0,
        "CASE_RATE_PER_ADMIT_LANG_AMT": None,
        "CASE_RATE_PER_ADMIT_LANG": None, "LER_LANG_IND": 0,
        "LER_LANG_TIMEFRAME": None, "LER_LANG": None, "TERMINATION_LANG_IND": 0,
        "TERMINATION_LANG_NOTICE_PERIOD": None, "TERMINATION_LANG": None,
        "MANUAL_ADM_LANG_IND": 0, "MANUAL_ADM_LANG_DATE": None,
        "MANUAL_ADM_LANG": None, "MEDICARE_INPATIENT_PPS_LANG_IND": 0,
        "MEDICARE_INPATIENT_PPS_LANG": None,
        "NLP_EXTRACTION_STATUS": "PROCESSING", "NLP_USER_ID": "SIV0592",
        "NLP_PROCESS_TIMESTAMP": None, "NLP_ERROR_COMMENTS": None
    }

# --- 5. Main Controller Function ---

def main_process_file(file_path: str, cis_ctrct_id: str, all_examples: dict):
    """
    Main controller to run the full, general-purpose pipeline.
    """
    
    results = initialize_all_fields()
    
    start_time = time.time()
    error_comments = ""
    status = "PROCESSING"

    try:
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_name)[1]
        
        # 1. (Part 1) Pre-process
        content_blocks = preprocess_document(file_path)
        if not content_blocks:
            raise ValueError("Document is empty or could not be processed.")

        # 2. (Part 2) Run LLM extraction
        extracted_data = extract_data_from_document(content_blocks, all_examples)
        
        # 3. Update results
        results.update(extracted_data)
        
        status = "COMPLETED"
        # If Pass 1 or 2 failed, the error will be in the extracted_data
        if results.get("NLP_ERROR_COMMENTS"):
             error_comments = results["NLP_ERROR_COMMENTS"]
             status = "FAILED"

    except Exception as e:
        status = "FAILED"
        error_comments = str(e)
        print(f"FATAL ERROR: {e}")
    
    finally:
        # 4. Add all metadata fields
        results['CIS_CTRCT_ID'] = cis_ctrct_id
        results['FILE_NAME'] = file_name
        results['FILE_EXTENSION'] = file_extension
        results['NLP_EXTRACTION_STATUS'] = status
        results['NLP_USER_ID'] = 'SIV0592'
        results['NLP_PROCESS_TIMESTAMP'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        results['NLP_ERROR_COMMENTS'] = error_comments

    return results

# --- 6. Example Execution ---

if __name__ == "__main__":
    
    # 1. Load examples ONCE
    # This MUST point to your Excel file
    ALL_EXAMPLES = load_all_examples('keywords_patters.xlsx')
    
    print("\n--- Starting Document Processing ---")
    
    # 2. --- SIMULATE A DOCUMENT ---
    # Create a dummy text file to test
    mock_file_path = "mock_contract.txt" 
    with open(mock_file_path, "w") as f:
        f.write("This is an AEC contract for General Hospital (NPI: 1234567890).\n")
        f.write("Attachment ID is Attach-123.\n")
        f.write("This Agreement may be modified by Plan with 90 days written notice.\n")
        f.write("CPT codes are covered. Re-hospitalization within 30 days is not covered.\n")

    # ❗ To test a real PDF, change this path:
    # mock_file_path = "path/to/your/scanned_document.pdf" 
    
    mock_cis_ctrct_id = "C-12345-ABC"
    
    # 3. Run the full pipeline
    if ALL_EXAMPLES: # Only run if examples were loaded
        final_data = main_process_file(
            file_path=mock_file_path,
            cis_ctrct_id=mock_cis_ctrct_id,
            all_examples=ALL_EXAMPLES
        )
        
        print("\n--- FINAL COMBINED OUTPUT ---")
        print(json.dumps(final_data, indent=2))
    else:
        print("\nSkipping processing because example file could not be loaded.")
