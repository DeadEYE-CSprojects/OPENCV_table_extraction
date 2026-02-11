def process_pipeline(start_index=None, end_index=None):
    
    # 1. Load Inventory
    df_inventory = get_or_create_inventory()
    
    if df_inventory.empty:
        print("No files to process. Exiting.")
        return

    # 2. Determine Processing Range
    if start_index is None: start_index = 0
    if end_index is None: end_index = len(df_inventory) - 1

    # Bounds check
    if start_index < 0: start_index = 0
    if end_index >= len(df_inventory): end_index = len(df_inventory) - 1

    print(f"--- Pipeline Started: Processing Index {start_index} to {end_index} (High-Res Mode) ---")
    if not RUN_CONTRACT_SCRIPT:
        print(">>> MODE: CONVERSION ONLY (Skipping Contract Scripts) <<<")

    index = start_index

    # --- HELPER: MEMORY SAFE GENERATOR ---
    def get_pdf_images_safely(pdf_path, dpi=500):
        """Yields images 10 at a time to save RAM."""
        import pdf2image
        try:
            info = pdf2image.pdfinfo_from_path(pdf_path)
            max_pages = info["Pages"]
        except:
            return # Handle non-PDFs or errors
            
        for i in range(1, max_pages + 1, 10):
            try:
                batch = pdf2image.convert_from_path(
                    pdf_path, dpi=dpi, first_page=i, last_page=min(i + 9, max_pages)
                )
                for page in batch:
                    yield page
            except Exception as e:
                print(f"      [Error] Batch conversion failed: {e}")
                break

    # 3. Main Loop
    while index <= end_index:
        try:
            row = df_inventory.iloc[index]
            f_name = row['filename']
            f_path = row['file_path']
            f_ext = row['file_ext']
            cis_id = row['CIS ID']
            
            clean_ext = f_ext.replace('.', '')
            final_txt_name = f"{f_name}_{clean_ext}.txt"
            final_txt_path = os.path.join(TXT_OUTPUT_PATH, final_txt_name)

            print(f"\n[{index}] Processing: {f_name} (CIS: {cis_id})")

            # --- A. SKIP CHECK ---
            if os.path.exists(final_txt_path):
                print(f"   -> Output found. Skipping extraction.")
                with open(final_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    txt_content = f.read()
                goto_step_5(f_name, clean_ext, cis_id, txt_content, f_path, final_txt_path)
                index += 1
                continue

            # --- B. FILE PROCESSING ---
            if not os.path.exists(f_path):
                print(f"   -> ERROR: File not found: {f_path}")
                log_process_status(f"MISSING FILE: {f_name}")
                index += 1
                continue

            # TYPE 1: SPREADSHEETS
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

            # TYPE 3: COMPLEX DOCUMENTS (PDF/DOCX/IMG)
            elif f_ext in ['.pdf', '.docx', '.tiff', '.tif', '.jpg', '.png', '.jpeg']:
                print("   -> Type: Complex Document (High Res)")
                temp_pdf_path = f_path
                is_docx = (f_ext == '.docx')
                
                if is_docx:
                    print("      -> Converting DOCX to PDF...")
                    temp_pdf_path = os.path.join(TXT_OUTPUT_PATH, f"temp_{cis_id}.pdf")
                    docx_to_pdf_convert(f_path, temp_pdf_path)

                total_extracted_text = ""
                
                # --- GENERATOR LOOP START (Replaces 'images = convert...') ---
                print("      -> Rasterizing PDF safely (Generator Mode)...")
                
                # Handle single images vs PDF generator
                if f_ext in ['.tiff', '.tif', '.jpg', '.png', '.jpeg']:
                    # Simple single image loading
                    img_iter = [Image.open(f_path)]
                else:
                    # PDF Generator
                    img_iter = get_pdf_images_safely(temp_pdf_path, dpi=PDF_CONVERSION_DPI)

                for i, pil_image in enumerate(img_iter):
                    page_num = i + 1
                    print(f"      -> Page {page_num}...")
                    
                    temp_img_path = f"temp_page_{page_num}.png"
                    pil_image.save(temp_img_path)
                    
                    # 1. Check Complexity
                    is_complex, token_cost = check_page_complexity(temp_img_path)
                    page_text = ""
                    current_tokens = token_cost

                    if is_complex:
                        print("         -> Complex (LLM High-Res).")
                        deskewed = deskew_image(pil_image)
                        final_img = enhance_and_upscale(deskewed)
                        
                        txt_llm, t_ocr = llm_convert_to_text(final_img)
                        page_text = txt_llm
                        current_tokens += t_ocr
                        log_token_usage_excel(f_name, current_tokens, 0)
                    else:
                        print("         -> Simple (Digital/OCR).")
                        digital_text = ""
                        has_digital = False
                        
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
                            # Fallback to Tesseract
                            raw_ocr = pytesseract.image_to_string(pil_image)
                            
                            # --- NEW: GEMINI CLEANING ---
                            print("            -> [Gemini] Cleaning OCR output...")
                            page_text = clean_ocr_text_gemini(raw_ocr)

                    # Append
                    formatted_page = f"\n##-- PAGE: {page_num} --##\n{page_text}\n"
                    total_extracted_text += formatted_page
                    
                    with open(final_txt_path, 'a', encoding='utf-8') as f: f.write(formatted_page)
                    
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)
                    
                    # Force Memory Release
                    pil_image.close()

                # Cleanup Temp PDF
                if is_docx and os.path.exists(temp_pdf_path): os.remove(temp_pdf_path)
                
                goto_step_5(f_name, clean_ext, cis_id, total_extracted_text, f_path, final_txt_path)

            else:
                print(f"   [Skipped] Unknown file type: {f_ext}")
                log_process_status(f"SKIPPED: {f_name}")

            index += 1

        except Exception as e:
            print(f"!!! ERROR on Index {index}: {e}")
            log_process_status(f"ERROR: Index {index} - {str(e)}")
            index += 1
            time.sleep(1)
            continue
