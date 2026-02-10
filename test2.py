# ==========================================
# REPLACEMENT FUNCTIONS
# ==========================================

def determine_contract_type(text_content):
    """
    Scans text for specific medical keywords to identify the contract type.
    Priority:
    1. If MULTIPLE types are detected -> Returns 'Others'
    2. If ONE type is detected -> Returns that type
    3. If NO types are detected -> Returns 'Others'
    """
    
    # Normalize text for matching
    text_upper = text_content.upper()
    
    # Define Regex Patterns for your 22 types
    # I have generated reasonable keywords. You can update these lists later.
    patterns = {
        "home_health": [r"HOME\s+HEALTH", r"HOME\s+CARE\s+AGENCY", r"\bHHA\b"],
        "skilled_nursing": [r"SKILLED\s+NURSING", r"\bSNF\b", r"NURSING\s+FACILITY"],
        "aec": [r"AMBULATORY\s+EMERGENCY", r"\bAEC\b", r"FREESTANDING\s+EMERGENCY"],
        "asc": [r"AMBULATORY\s+SURGERY", r"\bASC\b", r"SURGICAL\s+CENTER"],
        "detox": [r"DETOXIFICATION", r"\bDETOX\b", r"SUBSTANCE\s+ABUSE\s+TREATMENT"],
        "dialysis": [r"DIALYSIS", r"RENAL\s+DISEASE", r"\bESRD\b", r"KIDNEY\s+CENTER"],
        "hca": [r"HCA\s+HEALTHCARE", r"HOSPITAL\s+CORPORATION\s+OF\s+AMERICA"], # Assuming HCA Health System
        "hospice": [r"HOSPICE", r"PALLIATIVE\s+CARE", r"END\s+OF\s+LIFE"],
        "psych": [r"PSYCHIATRIC", r"MENTAL\s+HEALTH", r"BEHAVIORAL\s+HEALTH", r"\bIPF\b"],
        "rehab": [r"REHABILITATION", r"PHYSICAL\s+THERAPY", r"OCCUPATIONAL\s+THERAPY"],
        "tenet": [r"TENET\s+HEALTH", r"TENET\s+HOSPITAL"],
        "prosthetics": [r"PROSTHETICS", r"ORTHOTICS", r"\bP&O\b", r"DMEPOS"],
        "drg": [r"DIAGNOSIS\s+RELATED\s+GROUP", r"DRG\s+REIMBURSEMENT", r"MS-DRG"],
        "cah": [r"CRITICAL\s+ACCESS", r"\bCAH\b"],
        "rhc": [r"RURAL\s+HEALTH\s+CLINIC", r"\bRHC\b"],
        "dme": [r"DURABLE\s+MEDICAL\s+EQUIPMENT", r"\bDME\b", r"MEDICAL\s+SUPPLY"],
        "anesthesia": [r"ANESTHESIA", r"ANESTHESIOLOGY", r"CRNA"],
        "chv": [r"CARDIOLOGY", r"HEART\s+AND\s+VASCULAR", r"\bCHV\b"],
        "surgery": [r"GENERAL\s+SURGERY", r"SURGEON", r"OPERATIVE\s+SERVICES"], # General surgery (non-ASC)
        "telemedicine": [r"TELEMEDICINE", r"TELEHEALTH", r"REMOTE\s+PATIENT"],
        "audiology": [r"AUDIOLOGY", r"HEARING\s+AID", r"HEARING\s+TEST"]
    }

    detected_types = []

    # Scan for each type
    print("      -> [Classifier] Scanning for contract keywords...")
    for c_type, regex_list in patterns.items():
        for pattern in regex_list:
            if re.search(pattern, text_upper):
                if c_type not in detected_types:
                    detected_types.append(c_type)
                break # Move to next type once found

    # Logic for Classification
    if len(detected_types) > 1:
        print(f"      -> [Classifier] Multiple types found: {detected_types}. Defaulting to 'Others'.")
        return "Others"
    elif len(detected_types) == 1:
        return detected_types[0]
    else:
        print("      -> [Classifier] No specific keywords found. Defaulting to 'Others'.")
        return "Others"

def goto_step_5(filename, filetype, cis_id, text_content, original_path, txt_path):
    """
    Step 5: Finalize.
    Identifies the contract type using REGEX and runs the specific Python script.
    """
    if not RUN_CONTRACT_SCRIPT:
        print(f"   -> Conversion Complete. Saved to: {os.path.basename(txt_path)}")
        log_process_status(f"CONVERTED ONLY: {filename}")
        return
    
    print("   -> Step 5: Contract Analysis (Regex Mode)")
    
    # 1. Identify Type using Regex/Keywords (No LLM)
    ctype = determine_contract_type(text_content)
    print(f"      -> Final Classification: {ctype}")
    
    # 2. Locate the specific script
    # Standardize casing: home_health -> home_health.py, Others -> others.py
    target_script = f"{ctype.lower()}.py"
    script_full_path = os.path.join(CONTRACT_SCRIPTS_PATH, target_script)
    
    # 3. Fallback logic
    # If the specific script (e.g., 'home_health.py') doesn't exist, use 'others.py'
    # BUT pass the detected type as an argument so 'others.py' knows what it is.
    if not os.path.exists(script_full_path):
        print(f"      [Info] Script '{target_script}' not found. Falling back to 'others.py'.")
        script_full_path = os.path.join(CONTRACT_SCRIPTS_PATH, "others.py")
    
    # 4. Execute Script
    if os.path.exists(script_full_path):
        try:
            print(f"      -> Executing: {script_full_path}")
            subprocess.run([
                "python", script_full_path,
                "--filename", str(filename), 
                "--filetype", str(filetype), 
                "--cis_id", str(cis_id),
                "--contract_type", str(ctype),  # Pass the detected type (e.g., 'home_health')
                "--file_path", str(original_path), 
                "--txt_path", str(txt_path)
            ], check=True)
            log_process_status(f"SUCCESS: {filename} processed as {ctype}")
        except subprocess.CalledProcessError as e:
            print(f"      [Error] Subprocess Failed: {e}")
            log_process_status(f"ERROR: Script execution failed for {filename}")
    else:
        print(f"      [Error] CRITICAL: 'others.py' is missing in {CONTRACT_SCRIPTS_PATH}.")
        log_process_status(f"ERROR: Missing others.py for {filename}")
