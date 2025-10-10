import os
import cv2
import pytesseract
import numpy as np
import pandas as pd

# --- IMPORTANT SETUP ---
# Update this path if Tesseract is not in your system's PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def crop_qr_and_last_text(image):
    """Crops from QR code to last text element (for CMS1500, HealthInsuranceClaim)."""
    detector = cv2.QRCodeDetector()
    retval, points, _ = detector.detectAndDecode(image)
    if points is None:
        print("    - Error: QR Code not found.")
        return None
    x1, y1 = int(points[0][0][0]), int(points[0][0][1])

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    if not any(text.strip() for text in data['text']):
        print("    - Error: No text found.")
        return None
        
    last_element_index = np.argmax([t + h for t, h in zip(data['top'], data['height'])])
    x2 = data['left'][last_element_index] + data['width'][last_element_index]
    y2 = data['top'][last_element_index] + data['height'][last_element_index]
    
    return image[y1:y2, x1:x2]

def crop_hbcf(image):
    """Crops from 'Health Benefits Claim Form' text to the last grid lines."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    data['text'] = data['text'].str.lower().str.strip()
    
    phrase_data = data[data['text'].str.contains("health|benefits|claim|form", na=False)]
    if phrase_data.empty:
        print("    - Error: HBCF title text not found.")
        return None
    x1, y1 = phrase_data['left'].min(), phrase_data['top'].min()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        print("    - Error: No lines detected for HBCF cropping.")
        return None
        
    last_y = 0
    last_x = 0
    for line in lines:
        x_start, y_start, x_end, y_end = line[0]
        if abs(y_start - y_end) < 10: last_y = max(last_y, y_start, y_end)
        if abs(x_start - x_end) < 10: last_x = max(last_x, x_start, x_end)
            
    if last_x == 0 or last_y == 0:
        print("    - Error: Could not determine HBCF grid boundaries.")
        return None
        
    return image[y1:y2, x1:x2]

def crop_ub04(image):
    """Crops the main table/grid from a UB04 form."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    grid_mask = cv2.add(detected_horizontal, detected_vertical)
    contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("    - Error: No UB04 table contours found.")
        return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return image[y:y+h, x:x+w]

def crop_form(image_path, form_type):
    """Main router function to call the correct cropping logic."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"    - Error: Could not read image at {image_path}")
        return None

    form_type_lower = form_type.lower()
    
    if 'cms1500' in form_type_lower or 'healthinsuranceclaim' in form_type_lower:
        return crop_qr_and_last_text(image)
    elif 'hbcf' in form_type_lower:
        return crop_hbcf(image)
    elif 'ub04' in form_type_lower:
        return crop_ub04(image)
    else:
        print(f"    - Warning: Unknown form type '{form_type}' for filename. Skipping.")
        return None

# --- MAIN EXECUTION TO PROCESS A FOLDER ---
if __name__ == '__main__':
    # 1. CONFIGURE YOUR FOLDER PATHS
    input_folder = 'path/to/your/input_images'
    output_folder = 'path/to/your/output_folder'

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"--- Starting Cropping Process ---")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}\n")

    # 2. LOOP THROUGH ALL FILES IN THE INPUT FOLDER
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            full_image_path = os.path.join(input_folder, filename)
            print(f"Processing: {filename}...")

            # 3. DETERMINE FORM TYPE FROM FILENAME (e.g., "CMS1500_doc1.jpg")
            form_type_from_name = filename.split('_')[0]

            # 4. CROP THE IMAGE BASED ON ITS TYPE
            cropped_image = crop_form(full_image_path, form_type_from_name)

            # 5. SAVE THE CROPPED IMAGE TO THE OUTPUT FOLDER
            if cropped_image is not None:
                # Create a new filename for the output
                output_filename = f"cropped_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, cropped_image)
                print(f"  -> Successfully cropped and saved to {output_path}")
            else:
                print(f"  -> Cropping failed for this image.")
    
    print("\n--- Cropping Process Complete ---")
