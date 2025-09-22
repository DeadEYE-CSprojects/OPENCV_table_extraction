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

# --- 2. Define the Patterns to Check in Each Sentence ---

# Part A: Keywords to look for (readmission or preadmission)
keyword_pattern = r"(?:re-?|pre-?)admissions?"

# Part B: Time references to look for
time_value = r"(\d+|three|twenty-?four|forty-?eight|seventy-?two|seventy\s*two)"
time_unit = r"(hours?|hrs?|days?)"
timeframe_pattern = fr"{time_value}\s*{time_unit}"

# Part C: A single pattern to check if BOTH are in a sentence
# This looks for the keyword, followed by any text, followed by the time reference.
check_pattern = re.compile(
    fr".*?{keyword_pattern}.*?{timeframe_pattern}.*?",
    re.IGNORECASE
)

def format_excel_sheet(workbook_path):
    """Applies professional formatting to the generated Excel file."""
    try:
        wb = load_workbook(workbook_path)
        ws = wb.active
        header_font = Font(bold=True)
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        cell_alignment = Alignment(wrap_text=True, vertical='top', horizontal='left')

        for cell in ws[1]:
            cell.font = header_font
            cell.border = thin_border
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.border = thin_border
                cell.alignment = cell_alignment
        for col in ws.columns:
            max_length = 0
            column = get_column_letter(col[0].column)
            for cell in col:
                try:
                    lines = str(cell.value).split('\n')
                    max_length = max(max_length, max(len(line) for line in lines))
                except: pass
            adjusted_width = min((max_length + 2), 120)
            ws.column_dimensions[column].width = adjusted_width
        wb.save(workbook_path)
        print("\nExcel formatting applied successfully.")
    except Exception as e:
        print(f"\nCould not apply Excel formatting. Error: {e}")

def process_files():
    """
    Splits text into sentences and extracts those containing both a keyword and a time reference.
    """
    print(f"Starting analysis on files in: {INPUT_FOLDER}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    all_extracted_sentences = []
    
    txt_files = list(INPUT_FOLDER.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No TXT files found in {INPUT_FOLDER}")
        return

    for file_path in txt_files:
        print(f"-> Analyzing: {file_path.name}")
        
        try:
            full_text = file_path.read_text(encoding='utf-8', errors='ignore')
            # Normalize whitespace to make sentence splitting easier
            clean_text = re.sub(r'\s+', ' ', full_text).strip()
            # Split the text into sentences using periods as delimiters
            # This looks for a period followed by a space or the end of the text
            sentences = re.split(r'(?<=\.)\s*', clean_text)
        except Exception as e:
            print(f"  - Could not read file {file_path.name}: {e}")
            continue

        for sentence in sentences:
            if not sentence:
                continue
            
            # Check if the sentence contains BOTH the keyword and the time pattern
            if check_pattern.search(sentence):
                all_extracted_sentences.append({
                    "Filename": file_path.name,
                    "Extracted Sentence": sentence.strip()
                })

    if not all_extracted_sentences:
        print("\nProcessing complete. No matching sentences were found in any files.")
        return

    print(f"\nProcessing complete. Extracted {len(all_extracted_sentences)} sentences.")
    
    output_df = pd.DataFrame(all_extracted_sentences)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_excel_path = OUTPUT_FOLDER / f"Sentence_Extraction_{timestamp}.xlsx"
    
    output_df.to_excel(output_excel_path, index=False, engine='openpyxl')
    print(f"Results saved to: {output_excel_path}")
    
    format_excel_sheet(output_excel_path)

# --- Main execution block ---
if __name__ == "__main__":
    if not INPUT_FOLDER.exists():
        print(f"Error: The specified input folder does not exist: {INPUT_FOLDER}")
    else:
        process_files()
