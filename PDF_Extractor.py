import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, Alignment
import config
import patterns

def find_page_for_match(full_text, match_object):
    """Finds the page or sheet ID for a given regex match."""
    if not match_object:
        return ""
    # Find all page/sheet markers that appear *before* the match
    markers = re.findall(r'Start of (?:Page\s|Sheet:\s)(.*)', full_text[:match_object.start()])
    if markers:
        return markers[-1].strip()  # Return the last marker found
    return "1"  # Default to 1 if no markers are found before it

def create_excel_report():
    """
    Main function to analyze all .txt files and generate the final Excel report.
    """
    all_results = []

    # Create a lookup map for original file extensions
    print("Scanning source folder for file extensions...")
    extension_map = {}
    supported_patterns = ['*.pdf', '*.docx', '*.xlsx', '*.xls', '*.txt']
    for pattern in supported_patterns:
        for source_file in config.source_folder.glob(pattern):
            extension_map[source_file.stem] = source_file.suffix

    print(f"Starting analysis on files in: {config.INPUT_TXT_FOLDER}")

    for file_path in config.INPUT_TXT_FOLDER.glob('*.txt'):
        print(f" -> Analyzing: {file_path.name}")
        
        base_name = file_path.stem.replace('_Extracted', '')
        original_extension = extension_map.get(base_name, '.unknown')
        cis_id_match = re.search(r'(\d{6})', base_name)
        cis_id = cis_id_match.group(1) if cis_id_match else ""
        
        full_text = file_path.read_text(encoding='utf-8')

        # --- FOCUSED EXTRACTION LOGIC ---
        
        # Initialize default values for the final report row
        lang_present = "No"
        lang_phrase = ""
        page_num_str = ""
        associated_tin = ""
        lob_found = ""
        provider_name = ""

        # Step 1: Find the main language phrase. If not found, we skip the complex logic.
        lang_match1 = patterns.LANGUAGE_PART_1_PATTERN.search(full_text)
        lang_match2 = patterns.LANGUAGE_PART_2_PATTERN.search(full_text)
        
        if lang_match1 and lang_match2:
            lang_pos = lang_match2.start() # Position of the language phrase

            # Step 2: Find ALL potential Tax IDs that appear BEFORE the language phrase
            all_tin_matches_before = [m for m in patterns.TAX_ID_PATTERN.finditer(full_text) if m.start() < lang_pos]
            
            if all_tin_matches_before:
                # Find the one with the highest start position (the closest one)
                closest_tin_match = max(all_tin_matches_before, key=lambda m: m.start())
                
                # Step 3: Check for the required LOB between the TIN and the language
                search_area = full_text[closest_tin_match.end():lang_pos]
                lob_match = patterns.LOB_PATTERN.search(search_area)
                
                # THE CRITICAL FILTER: Only proceed if the LOB is found (e.g., "Medicare")
                if lob_match:
                    # If all conditions are met, populate all the variables
                    lang_present = "Yes"
                    lang_phrase = f"{lang_match1.group(0).strip()} | {lang_match2.group(0).strip()}"
                    associated_tin = closest_tin_match.group(1)
                    lob_found = lob_match.group(1).strip()
                    
                    page_of_lang = find_page_for_match(full_text, lang_match2)
                    page_of_tin = find_page_for_match(full_text, closest_tin_match)
                    page_num_str = f"Lang on: {page_of_lang}, TIN on: {page_of_tin}"

                    # Step 4: Search the entire document for a general provider name
                    provider_match = patterns.PROVIDER_NAME_PATTERN.search(full_text)
                    if provider_match:
                        provider_name = provider_match.group(1).strip()

        # --- Assemble the final record for this file ---
        all_results.append({
            'FileName': base_name,
            'CIS_id': cis_id,
            'File_Extension': original_extension.replace('.', ''),
            'Lang_present': lang_present,
            'Lang_phrase': lang_phrase,
            'Page_num or sheet_name': page_num_str,
            'Associated_TIN': associated_tin,
            'LOB': lob_found,
            'Provider/Facility Name': provider_name,
            'NLP_Status': 'Completed',
            'NLP_Id': 'SIV 0592',
            'NLP_RUN_DATETIME': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    if not all_results:
        print("Warning: No data extracted. Excel file will not be created.")
        return

    # Create the DataFrame and save the formatted Excel file
    print(f"\nAnalysis complete. Creating Excel report...")
    column_order = [
        'FileName', 'CIS_id', 'File_Extension', 'Lang_present', 'Lang_phrase', 
        'Page_num or sheet_name', 'Associated_TIN', 'LOB', 'Provider/Facility Name',
        'NLP_Status', 'NLP_Id', 'NLP_RUN_DATETIME'
    ]
    df = pd.DataFrame(all_results)
    df = df.reindex(columns=column_order)
    
    with pd.ExcelWriter(config.OUTPUT_EXCEL_PATH, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Contract Analysis', index=False)
        worksheet = writer.sheets['Contract Analysis']
        for col_idx, column_cells in enumerate(worksheet.columns, 1):
            header_cell = worksheet.cell(row=1, column=col_idx)
            header_cell.font = Font(bold=True)
            max_length = max(len(str(cell.value)) for cell in column_cells)
            worksheet.column_dimensions[get_column_letter(col_idx)].width = min(max_length + 2, 60)
            for cell in column_cells[1:]:
                cell.alignment = Alignment(wrap_text=True, vertical='top')

    print(f"✅ Excel report created successfully at: {config.OUTPUT_EXCEL_PATH}")

if __name__ == "__main__":
    create_excel_report()