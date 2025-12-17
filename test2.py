def run_complex_rate_logic(xls, sheet_name):
    clean_data = []
    try:
        # 1. FIND HEADER ROW
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=30)
        
        header_idx = 0
        found_header = False
        
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(row.astype(str)).lower()
            if "facility name" in row_str:
                header_idx = idx
                found_header = True
                break
        
        if not found_header:
            return pd.read_excel(xls, sheet_name=sheet_name) 

        # 2. LOAD METADATA (ROWS ABOVE HEADER)
        meta_rows = []
        if header_idx > 0:
            start_row = max(0, header_idx - 4) 
            meta_df = pd.read_excel(xls, sheet_name=sheet_name, header=None, 
                                    skiprows=start_row, nrows=header_idx - start_row)
            # Convert to list of lists for safe indexing
            meta_rows = meta_df.values.tolist()

        # 3. LOAD MAIN DATA
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header_idx)
        df.columns = df.columns.astype(str).str.strip()

        # 4. IDENTIFY COLUMNS
        static_keywords = ['facility', 'tin', 'city', 'market']
        static_cols = []
        data_cols = []
        
        for col in df.columns:
            if any(k in col.lower() for k in static_keywords):
                static_cols.append(col)
            else:
                data_cols.append(col)

        # 5. PROCESS ROW BY ROW
        for idx, row in df.iterrows():
            fac_col = next((c for c in static_cols if 'facility' in c.lower()), None)
            fac_name = row[fac_col] if fac_col else None

            if pd.isna(fac_name): continue

            base_info = {
                'Facility Name': fac_name,
                'City': row.get(next((c for c in static_cols if 'city' in c.lower()), ''), ''),
                'TIN': row.get(next((c for c in static_cols if 'tin' in c.lower()), ''), ''),
                'LOB': 'Medicare'
            }
            
            for col_name in data_cols:
                cell_value = row[col_name]
                if pd.isna(cell_value): continue

                # --- FIND DATE & RM (Robust Indexing) ---
                # Safe way to get integer index even if columns are duplicates
                try:
                    col_loc = df.columns.get_loc(col_name)
                    # If multiple columns have same name, get_loc returns a slice or array
                    # We take the first one to avoid "list indices must be int" error
                    if isinstance(col_loc, slice):
                        col_idx = col_loc.start
                    elif isinstance(col_loc, np.ndarray) or isinstance(col_loc, list):
                        col_idx = np.where(col_loc)[0][0] # First True index
                    else:
                        col_idx = int(col_loc)
                except:
                    col_idx = 0 # Fallback safety

                found_date = "Unknown"
                found_rm = "Unknown"

                # Check rows ABOVE the header
                if meta_rows:
                    for m_row in reversed(meta_rows):
                        try:
                            val = str(m_row[col_idx]).strip()
                        except: continue # Skip if index out of bounds

                        if val == 'nan' or val == '': continue
                        
                        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', val)
                        if date_match:
                            found_date = date_match.group(1)
                        elif len(val) > 2 and "effective" not in val.lower(): 
                            found_rm = val

                if found_date == "Unknown":
                    header_date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', str(col_name))
                    if header_date_match:
                        found_date = header_date_match.group(1)

                # --- PARSE CELL VALUE (FIXED) ---
                str_val = str(cell_value).strip()
                
                # 1. STRICT MATCH: Look for "PPO:" or "HMO:"
                matches = re.findall(r'(PPO|HMO)\s*:\s*([^\n\r]+)', str_val, re.IGNORECASE)
                
                # 2. FALLBACK: If no PPO/HMO found, use "Standard"
                if not matches:
                    if str_val and str_val.lower() != 'nan':
                        # Valid List of Tuples: [ (Plan, Rate) ]
                        matches = [('Standard', str_val)]

                # 3. Create Rows (Safe Unpacking)
                for item in matches:
                    # SAFETY: Ensure we have exactly 2 items to unpack
                    if len(item) != 2: 
                        continue

                    plan, raw_rate = item
                    clean_rate = str(raw_rate).replace('Medicare', '').replace('medicare', '').strip()
                    
                    new_row = base_info.copy()
                    new_row['RM'] = found_rm
                    new_row['Effective Date'] = found_date
                    new_row['Plan'] = str(plan).upper()
                    new_row['Rate'] = clean_rate
                    clean_data.append(new_row)

        if clean_data:
            final_df = pd.DataFrame(clean_data)
            desired = ['Facility Name', 'City', 'TIN', 'LOB', 'RM', 'Effective Date', 'Plan', 'Rate']
            exist_cols = [c for c in desired if c in final_df.columns]
            return final_df[exist_cols]
        else:
            return pd.read_excel(xls, sheet_name=sheet_name)

    except Exception as e:
        print(f"Error in logic for {sheet_name}: {e}")
        return pd.read_excel(xls, sheet_name=sheet_name)
