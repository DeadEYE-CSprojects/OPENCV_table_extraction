import pandas as pd
import re
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

# --- 1. LOGIC FUNCTIONS ---

def run_complex_rate_logic(xls, sheet_name):
    """
    Scans for 'Facility Name', extracts methodology, and parses PPO/HMO rates via Regex.
    Returns a processed DataFrame.
    """
    clean_data = []
    
    # A. FIND HEADER ROW (Scanning for 'Facility Name')
    try:
        df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None, nrows=20)
        
        header_idx = 0
        found_header = False
        for idx, row in df_raw.iterrows():
            row_str = ' '.join(row.astype(str)).lower()
            if "facility name" in row_str:
                header_idx = idx
                found_header = True
                break
        
        if not found_header:
            return pd.DataFrame() # Return empty if header not found

        # B. LOAD DATA & METHODOLOGY
        df = pd.read_excel(xls, sheet_name=sheet_name, header=header_idx)
        
        # Methodology is 2 rows ABOVE the header
        methodology_map = {}
        if header_idx >= 2:
            meta_row = pd.read_excel(xls, sheet_name=sheet_name, header=None, 
                                   skiprows=header_idx-2, nrows=1).iloc[0]
            for i, col_name in enumerate(df.columns):
                if i < len(meta_row):
                    methodology_map[col_name] = meta_row[i]

        # C. IDENTIFY COLUMNS
        df.columns = df.columns.astype(str).str.strip()
        static_keywords = ['facility', 'tin', 'city', 'market']
        static_cols = []
        date_cols = []
        
        for col in df.columns:
            if any(k in col.lower() for k in static_keywords):
                static_cols.append(col)
            else:
                date_cols.append(col)

        # D. EXTRACT & CLEAN ROWS
        for idx, row in df.iterrows():
            fac_col = next((c for c in static_cols if 'facility' in c.lower()), None)
            fac_name = row[fac_col] if fac_col else None

            if pd.isna(fac_name): continue

            base_info = {
                'Facility Name': fac_name,
                'City': row.get(next((c for c in static_cols if 'city' in c.lower()), ''), ''),
                'TIN': row.get(next((c for c in static_cols if 'tin' in c.lower()), ''), ''),
                'LOB': 'Medicare', 
                'RM': 'Unknown' 
            }
            
            for date_col in date_cols:
                cell_value = row[date_col]
                if pd.isna(cell_value): continue
                
                # Clean Header
                clean_date = str(date_col).split('\n')[0].replace('except if noted', '').strip()
                
                # Get Methodology
                meth_val = methodology_map.get(date_col)
                current_rm = meth_val if pd.notna(meth_val) else "Unknown"

                # Parse Cell: PPO/HMO and Rate
                str_val = str(cell_value)
                matches = re.findall(r'(PPO|HMO)[:\s-]+([^\n\r]+)', str_val, re.IGNORECASE)
                
                for plan, raw_rate in matches:
                    clean_rate = raw_rate.replace('Medicare', '').replace('medicare', '').strip()
                    
                    new_row = base_info.copy()
                    new_row['RM'] = current_rm
                    new_row['Effective Date'] = clean_date
                    new_row['Plan'] = plan.upper()
                    new_row['Rate'] = clean_rate
                    clean_data.append(new_row)

        # E. CREATE DATAFRAME
        if clean_data:
            final_df = pd.DataFrame(clean_data)
            desired_order = ['Facility Name', 'City', 'TIN', 'LOB', 'RM', 'Effective Date', 'Plan', 'Rate']
            # Reindex handling missing columns safely
            final_df = final_df.reindex(columns=desired_order)
            return final_df
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"Error in custom logic: {e}")
        return pd.DataFrame()

# --- 2. FORMATTING HELPER ---

def format_worksheet(worksheet, sheet_name):
    """
    Converts data to an Excel Table, auto-adjusts width, and enables text wrapping.
    """
    max_row = worksheet.max_row
    max_col = worksheet.max_column
    
    if max_row < 2: return # Skip empty sheets

    ref = f"A1:{get_column_letter(max_col)}{max_row}"
    clean_name = sheet_name.replace(" ", "_").replace("-", "_")
    
    # Ensure unique table name if needed, usually sheet name is unique enough
    tab = Table(displayName=f"Table_{clean_name}", ref=ref)
    style = TableStyleInfo(name="TableStyleMedium9", showFirstColumn=False,
                           showLastColumn=False, showRowStripes=True, showColumnStripes=False)
    tab.tableStyleInfo = style
    worksheet.add_table(tab)

    for i, column in enumerate(worksheet.columns):
        column_letter = get_column_letter(i + 1)
        max_length = 0
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except: pass
            cell.alignment = Alignment(wrap_text=True, vertical='center')

        adjusted_width = min(max_length + 2, 50) 
        worksheet.column_dimensions[column_letter].width = adjusted_width

# --- 3. MAIN EXECUTION ---

INPUT_FILE = 'Master_Rates.xlsx'
OUTPUT_FILE = 'Final_Output.xlsx'

xls = pd.ExcelFile(INPUT_FILE)

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    
    for sheet_name in xls.sheet_names:
        
        # Default read (overwritten if specific logic applies)
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        n = sheet_name
        
        # --- LOGIC SWITCH ---
        if n == 'Sheet1': 
            # Replace 'Sheet1' with the actual sheet name you want to process with the complex logic
            df = run_complex_rate_logic(xls, sheet_name)
            
        elif n == 'Another_Sheet_Name':
            # Add next logic here
            pass
        # --------------------

        # If data exists, write it
        if not df.empty:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Formatting
            worksheet = writer.sheets[sheet_name]
            format_worksheet(worksheet, sheet_name)
        else:
            print(f"Skipping empty or failed sheet: {sheet_name}")

print(f"Done. Saved to {OUTPUT_FILE}")