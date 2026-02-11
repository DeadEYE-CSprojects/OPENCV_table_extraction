# ==========================================
# 4. SINGLE FILE FILTERING UTILS
# ==========================================

def parse_page_ranges(range_string):
    """
    Parses "1-4, 49" into [0, 1, 2, 3, 48].
    """
    pages = set()
    parts = range_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            # Adjust 1-based input to 0-based Python index
            pages.update(range(start - 1, end)) 
        else:
            pages.add(int(part) - 1)
            
    return sorted(list(pages))

def create_filtered_temp_file(target_filename, page_range_str):
    """
    Slices the specific file and saves it to a temp folder.
    Returns the path to that temp folder.
    """
    # Define Paths
    source_path = os.path.join(INPUT_FILES_PATH, target_filename)
    temp_dir = "/dbfs/tmp/single_process_contract_ocr/" # Use DBFS temp path for Databricks
    
    # Clean/Recreate Temp Directory
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    if not os.path.exists(source_path):
        print(f"ERROR: Target file '{target_filename}' not found in {INPUT_FILES_PATH}")
        return None

    print(f"--- [Single File Mode] Filtering {target_filename} (Pages: {page_range_str}) ---")

    # A. Handle File Type (DOCX -> PDF)
    ext = os.path.splitext(target_filename)[1].lower()
    working_pdf_path = source_path
    
    # Note: docx2pdf might need LibreOffice installed in Databricks environment
    # If it fails, we assume PDF input for now.
    if ext == '.docx':
        print("   -> Converting DOCX to temp PDF...")
        converted_pdf = os.path.join(temp_dir, "temp_conversion.pdf")
        try:
            docx_to_pdf_convert(source_path, converted_pdf)
            working_pdf_path = converted_pdf
        except:
            print("   [Warning] DOCX conversion failed. Using original file if possible.")

    # B. Extract Pages using PyMuPDF (fitz)
    try:
        doc = fitz.open(working_pdf_path)
        selected_pages = parse_page_ranges(page_range_str)
        
        new_doc = fitz.open()
        
        for page_idx in selected_pages:
            if page_idx < len(doc):
                new_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            else:
                print(f"   [Warning] Page {page_idx + 1} out of range. Skipping.")
        
        # Save the Filtered File (Keep original name for CIS ID logic)
        final_output_path = os.path.join(temp_dir, os.path.splitext(target_filename)[0] + ".pdf")
        new_doc.save(final_output_path)
        new_doc.close()
        doc.close()
        
        print(f"   -> Created filtered file: {final_output_path}")
        return temp_dir

    except Exception as e:
        print(f"   [Error] Failed to slice PDF: {e}")
        return None



# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- 1. DEFINE WIDGETS ---
    try:
        dbutils.widgets.text("start_index", "0", "Start Index")
        dbutils.widgets.text("end_index", "100", "End Index")
        dbutils.widgets.text("target_filename", "", "Specific File (Optional)")
        dbutils.widgets.text("target_pages", "", "Page Range (e.g. 1-3, 50)")
    except:
        pass # Ignore if widgets already exist

    # --- 2. GET VALUES ---
    s_val = dbutils.widgets.get("start_index")
    e_val = dbutils.widgets.get("end_index")
    
    target_file = dbutils.widgets.get("target_filename").strip()
    target_pages = dbutils.widgets.get("target_pages").strip()

    start = int(s_val) if s_val.strip() else 0
    end = int(e_val) if e_val.strip() else None

    # --- 3. DECISION LOGIC ---
    
    if target_file and target_pages:
        # >>> MODE A: SINGLE FILE FILTERED <<<
        print(f"\n>>> ACTIVATING SINGLE FILE MODE: {target_file} (Pages: {target_pages}) <<<")
        
        # 1. Create the specific temp input folder
        new_input_dir = create_filtered_temp_file(target_file, target_pages)
        
        # 2. Define a temp output folder
        temp_output_dir = "/dbfs/tmp/single_process_output/"
        if os.path.exists(temp_output_dir): shutil.rmtree(temp_output_dir)
        os.makedirs(temp_output_dir)
        
        if new_input_dir:
            # Backup original global variables
            original_input_path = INPUT_FILES_PATH
            original_inventory_path = INVENTORY_FILE_PATH
            original_output_path = TXT_OUTPUT_PATH  # <--- Backup Original Output Path
            
            try:
                # 3. OVERRIDE Globals
                INPUT_FILES_PATH = new_input_dir
                INVENTORY_FILE_PATH = os.path.join(new_input_dir, "temp_inventory.xlsx")
                TXT_OUTPUT_PATH = temp_output_dir   # <--- Override Output Path
                
                print(f"--- Temporary Output Path: {TXT_OUTPUT_PATH} ---")

                # 4. RUN PIPELINE (Index 0)
                process_pipeline(start_index=0, end_index=0)
                
                print(f"\n>>> Single File Processing Complete. <<<")
                print(f">>> You can check results in: {temp_output_dir}")
                
            except Exception as e:
                print(f"!!! ERROR during Single File Mode: {e}")
                
            finally:
                # 5. CLEANUP
                print(f"--- Cleaning up temp directories... ---")
                
                # Delete Input Temp
                if os.path.exists(new_input_dir):
                    shutil.rmtree(new_input_dir)
                    print(f"   -> Deleted Input Temp: {new_input_dir}")
                
                # Delete Output Temp (Optional: Comment this out if you want to inspect results before deleting)
                # if os.path.exists(temp_output_dir):
                #    shutil.rmtree(temp_output_dir)
                #    print(f"   -> Deleted Output Temp: {temp_output_dir}")
                
                # Restore original paths
                INPUT_FILES_PATH = original_input_path
                INVENTORY_FILE_PATH = original_inventory_path
                TXT_OUTPUT_PATH = original_output_path
                print("--- Cleanup Complete. Restored original paths. ---")

    else:
        # >>> MODE B: STANDARD BATCH <<<
        print(f"\n>>> ACTIVATING BATCH MODE: Index {start} to {end} <<<")
        process_pipeline(start_index=start, end_index=end)
