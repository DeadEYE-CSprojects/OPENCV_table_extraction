import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from openpyxl import load_workbook
from openpyxl.styles import Font, Border, Side, Alignment
from openpyxl.utils import get_column_letter

# --- 1. Configuration ---
# TODO: Update these paths to your actual input and output folders.
INPUT_FOLDER = Path(r"C:\path\to\your\input_folder")
OUTPUT_FOLDER = Path(r"C:\path\to\your\output_folder")

# --- 2. Define the Regex Pattern in Parts ---

time_value = r"(\d+|one|two|three|twenty-?four|forty-?eight|seventy-?two|seventy\s*two)"
time_unit = r"(hours?|hrs?|days?)"
timeframe_pattern = fr"(?:within\s*(?:a\s*)?\(?\s*{time_value}\s*\)?\s*{time_unit})"

readmission_keyword = r"(?:re-?admit(?:ted|sions?))"
location_keyword = r"(?:to\s+the\s+)?(?:hospital|facility|pho|provider)"
reason_keyword = r"for\s*(?:the\s*)?same\s*(?:specific\s+)?(?:condit\w*|episode\s*of\s*care)"

# --- 3. Assemble the Final, Flexible Regex ---
final_pattern = re.compile(
    fr"{readmission_keyword}.*?{timeframe_pattern}.*?{reason_keyword}",
    re.IGNORECASE | re.DOTALL
)

def format_excel_sheet(workbook_path):
    """Applies professional formatting to the generated Excel file."""
    try:
        wb = load_workbook(workbook_path)
        ws = wb.active

        # Define styles
        header_font = Font(bold=True, name='Calibri', size=11)
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        # Vertical alignment is key for multi-line cells
        cell_alignment = Alignment(wrap_text=True, vertical='top', horizontal='left')

        # Apply formatting to header
        for cell in ws[1]:
            cell.font = header_font
            cell.border = thin_border

        # Apply formatting to all data cells
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row, max_col=ws.max_column):
            for cell in row:
                cell.border = thin_border
                cell.alignment = cell_alignment
        
        # Auto-fit columns
        for col in ws.columns:
            max_length = 0
            column = get_column_letter(col[0].column)
            # For multi-line cells, width fitting is based on the longest line
            for cell in col:
                try:
                    # Check each line in the cell for the max length
                    cell_lines = str(cell.value).split('\n')
                    for line in cell_lines:
                        if len(line) > max_length:
                            max_length = len(line)
                except:
                    pass
            # Set a practical limit on width to avoid extremely wide columns
            adjusted_width = min((max_length + 2), 100)
            ws.column_dimensions[column].width = adjusted_width

        wb.save(workbook_path)
        print("\nExcel formatting applied successfully.")

    except Exception as e:
        print(f"\nCould not apply Excel formatting. Error: {e}")


def process_files():
    """
    Processes all TXT files, finds ALL matching clauses in each,
    and saves the structured data to a formatted Excel file.
    """
    print("Starting file processing...")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    results_data = []
    
    txt_files = list(INPUT_FOLDER.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No TXT files found in {INPUT_FOLDER}")
        return

    for file_path in txt_files:
        print(f"Processing file: {file_path.name}")
        
        try:
            full_text = file_path.read_text(encoding='utf-8', errors='ignore')
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
        except Exception as e:
            print(f"  - Could not read file {file_path.name}: {e}")
            continue

        # Use finditer to find ALL matches in the text
        matches = list(final_pattern.finditer(clean_text))
        
        lang_present = "No"
        all_phrases = ""
        all_timeframes = ""

        if matches:
            lang_present = "Yes"
            phrases_found = []
            timeframes_found = []
            
            for match in matches:
                # Extract details for each match
                phrases_found.append(match.group(0).strip())
                
                time_value_extracted = match.group(1).strip()
                time_unit_extracted = match.group(2).strip().lower()
                timeframes_found.append(f"{time_value_extracted} {time_unit_extracted}")

            # Join all findings with two line gaps
            all_phrases = "\n\n".join(phrases_found)
            all_timeframes = "\n\n".join(timeframes_found)

        # Append one row per file with all required fields
        results_data.append({
            "filename": file_path.stem,
            "extension": file_path.suffix,
            "lang_present": lang_present,
            "lang_phrase": all_phrases,
            "Time frame": all_timeframes,
            "NLP id": "SEV9999",
            "NLP status": "completed"
        })

    if not results_data:
        print("\nProcessing complete. No relevant data found in any files.")
        return

    print(f"\nProcessing complete. Data extracted from {len(results_data)} files.")
    
    # --- 4. Save and Format the Excel File ---
    output_df = pd.DataFrame(results_data)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_excel_path = OUTPUT_FOLDER / f"Readmission_Extraction_{timestamp}.xlsx"
    
    output_df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"Results have been saved to: {output_excel_path}")
    
    format_excel_sheet(output_excel_path)


# --- Main execution block ---
if __name__ == "__main__":
    if not INPUT_FOLDER.exists():
        print(f"Error: The specified input folder does not exist: {INPUT_FOLDER}")
        print("Please update the 'INPUT_FOLDER' variable in the script.")
    else:
        process_files()
