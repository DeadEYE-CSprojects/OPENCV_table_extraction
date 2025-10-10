import cv2
import pytesseract
import numpy as np

# Make sure to configure your Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_checked_box(image_path, option1='yes', option2='no'):
    """
    Detects which checkbox is marked ('YES' or 'NO') in an image.

    Args:
        image_path (str): The path to the image file.
        option1 (str): The text of the first option (e.g., 'yes').
        option2 (str): The text of the second option (e.g., 'no').

    Returns:
        str: The selected option's text, 'Uncertain', or 'Error'.
    """
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found"

    # Use pytesseract to get detailed data about text, including bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    locations = {option1: None, option2: None}
    
    # Step 1: Find the locations of the option words ("YES" and "NO")
    for i in range(len(data['text'])):
        text = data['text'][i].lower().strip()
        if text == option1:
            locations[option1] = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        elif text == option2:
            locations[option2] = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])

    if not locations[option1] or not locations[option2]:
        return "Error: Could not find 'YES' and 'NO' text options."

    scores = {option1: 0, option2: 0}

    # Step 2: Analyze the checkbox region for each option
    for option, loc in locations.items():
        if loc:
            x, y, w, h = loc
            
            # Define the checkbox ROI. IMPORTANT: This assumes the box is to the LEFT
            # of the text. You may need to adjust these values for your specific form.
            box_width = h + 10  # Assume the box is roughly square, based on text height
            box_x = x - box_width - 5 # Position the box 5 pixels to the left of the text
            box_y = y - 5           # Align the box vertically with a small offset
            box_height = h + 10
            
            # Crop the checkbox ROI from the original image
            checkbox_roi = image[box_y:box_y + box_height, box_x:box_x + box_width]
            
            if checkbox_roi.size == 0:
                continue

            # Step 3: Count the dark pixels to determine if it's checked
            gray_roi = cv2.cvtColor(checkbox_roi, cv2.COLOR_BGR2GRAY)
            # Threshold to make it pure black and white
            _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate the percentage of black pixels (the mark)
            black_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            pixel_percentage = (black_pixels / total_pixels) * 100
            scores[option] = pixel_percentage
            
            # print(f"Analyzing '{option}' box... Found {pixel_percentage:.2f}% black pixels.")
            
    # Step 4: Compare scores and decide
    # A checked box should have a significantly higher percentage (e.g., > 5%)
    # An empty box should be very low (e.g., < 2%)
    if scores[option1] > 5 and scores[option2] < 2:
        return option1.upper()
    elif scores[option2] > 5 and scores[option1] < 2:
        return option2.upper()
    else:
        return "Uncertain"


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Replace with the path to your image
    image_file_path = 'path/to/your/20251010_153030 (1).jpg'
    
    result = detect_checked_box(image_file_path)
    
    print(f"\nResult: The selected option is '{result}'.")
