def run_stage_2_extraction():
    """Converts PNG -> Dynamic Cell Crops -> Excel Report."""
    print("\n--- STAGE 2: Extraction (PNG to Data) ---")
    os.makedirs(FINAL_OUTPUT_FOLDER, exist_ok=True)
    
    files = glob.glob(os.path.join(INTERMEDIATE_PNG_FOLDER, '*.png'))
    if not files:
        print("No PNG files found to extract.")
        return

    all_data = []
    
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        base_name = os.path.splitext(fname)[0]
        print(f"Extracting {i+1}/{len(files)}: {fname}")
        
        full_img = cv2.imread(fpath)
        if full_img is None: continue
        
        # Get dimensions for "bottom right most end" logic
        h, w = full_img.shape[:2]
            
        # Setup subfolder
        subfolder = os.path.join(FINAL_OUTPUT_FOLDER, base_name)
        os.makedirs(subfolder, exist_ok=True)
        
        row_data = {'Filename': fname}
        
        # =========================================================
        # DYNAMIC COORDINATE LOGIC
        # =========================================================
        
        # Check if file is Grid or NGrid
        is_grid = fname.startswith("Grid_")
        
        # --- Define Region 1 (Bottom Area) ---
        # Logic: Start at 21, 3111 and go to the absolute bottom-right (w, h)
        # "if non grid same" -> implies logic applies to both
        r1_coords = (21, 3111, w, h)
        
        # --- Define Region 2 (Specific Block) ---
        if is_grid:
            # Grid Coordinates
            r2_coords = (1927, 2111, 2415, 3051)
        else:
            # Non-Grid Coordinates
            r2_coords = (1971, 2261, 2455, 3000)

        # Pack into a temporary config list for processing
        current_file_config = {
            'Region_1': {'coords': r1_coords, 'method': 'default'},
            'Region_2': {'coords': r2_coords, 'method': 'default'}
        }
        
        # =========================================================
        # PROCESSING LOOP
        # =========================================================
        for key, conf in current_file_config.items():
            x1, y1, x2, y2 = conf['coords']
            method = conf['method']
            
            # Safety Bounds: Ensure we don't crop outside the image
            # "pixels will be out of range... avoid them and get end most point"
            safe_x1 = max(0, x1)
            safe_y1 = max(0, y1)
            safe_x2 = min(w, x2)
            safe_y2 = min(h, y2)
            
            # Sanity Check: valid crop area
            if safe_x2 > safe_x1 and safe_y2 > safe_y1:
                crop = full_img[safe_y1:safe_y2, safe_x1:safe_x2]
                
                # Save Crop
                cv2.imwrite(os.path.join(subfolder, f"{key}.png"), crop)
                
                # Extract
                text = extract_data_from_cell(crop, method)
                row_data[key] = text
            else:
                print(f"   -> Warning: Invalid coords for {key} in {fname}")
                row_data[key] = "ERROR_COORDS"
            
        all_data.append(row_data)
        
    # Save Excel
    if all_data:
        df = pd.DataFrame(all_data)
        out_path = os.path.join(FINAL_OUTPUT_FOLDER, 'Final_Report.xlsx')
        df.to_excel(out_path, index=False)
        print(f"\nSUCCESS! Report saved to: {out_path}")
        print(df.head())
