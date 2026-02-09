def log_token_usage_excel(filename, input_tokens_p1, output_tokens_p1, status=0):
    excel_path = "inventory.xlsx"
    
    # --- PRICING CONSTANTS (Per Million) ---
    PRICE_INPUT  = 2.12
    PRICE_OUTPUT = 8.47
    
    # 1. Calculate Phase 1 Costs
    ip_cost_p1 = (input_tokens_p1 / 1_000_000) * PRICE_INPUT
    op_cost_p1 = (output_tokens_p1 / 1_000_000) * PRICE_OUTPUT
    
    # 2. Phase 2 Defaults (Set to 0 for now)
    ip_tokens_p2 = 0
    op_tokens_p2 = 0
    ip_cost_p2   = 0.0
    op_cost_p2   = 0.0
    
    # 3. Calculate Total Row Cost (P1 + P2)
    total_row_cost = ip_cost_p1 + op_cost_p1 + ip_cost_p2 + op_cost_p2

    # 4. Load or Create DataFrame
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        # Create with specific column order if file doesn't exist
        df = pd.DataFrame(columns=[
            'filename', 
            'ip_tokens_p1', 'ip_cost_p1', 'op_tokens_p1', 'op_cost_p1',
            'ip_tokens_p2', 'ip_cost_p2', 'op_tokens_p2', 'op_cost_p2',
            'Total_Cost', 'processed_flag'
        ])

    # 5. Ensure all columns exist (in case we run on an old Excel file)
    required_cols = [
        'ip_tokens_p1', 'ip_cost_p1', 'op_tokens_p1', 'op_cost_p1',
        'ip_tokens_p2', 'ip_cost_p2', 'op_tokens_p2', 'op_cost_p2',
        'Total_Cost'
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 6. Update the specific row
    mask = df['filename'] == filename
    if mask.any():
        # Update P1 Data
        df.loc[mask, 'ip_tokens_p1'] = input_tokens_p1
        df.loc[mask, 'ip_cost_p1']   = round(ip_cost_p1, 6)
        df.loc[mask, 'op_tokens_p1'] = output_tokens_p1
        df.loc[mask, 'op_cost_p1']   = round(op_cost_p1, 6)
        
        # Initialize P2 Data (Only if empty, to avoid overwriting future P2 runs)
        # But since you said "keep as 0 for now", we ensure they are 0.
        df.loc[mask, 'ip_tokens_p2'] = ip_tokens_p2
        df.loc[mask, 'ip_cost_p2']   = ip_cost_p2
        df.loc[mask, 'op_tokens_p2'] = op_tokens_p2
        df.loc[mask, 'op_cost_p2']   = op_cost_p2
        
        # Update Grand Total
        df.loc[mask, 'Total_Cost']   = round(total_row_cost, 6)
        df.loc[mask, 'processed_flag'] = status



import numpy as np
import cv2
from PIL import Image

def deskew_image(pil_image):
    """
    Deskews an image using projection profiles (User's Verified Logic).
    Wrapper handles PIL -> OpenCV -> PIL conversion.
    """
    if pil_image is None: return None
    
    # 1. Convert PIL (RGB) to OpenCV (BGR)
    image = np.array(pil_image)
    # Check if we need to convert color space
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        
    h, w = image.shape[:2]

    # 2. Downscale for speed (Processing logic)
    scale = 800 / max(h, w)
    # Avoid upscaling if image is already small
    if scale < 1:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = image

    # 3. Create Binary Threshold
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 4. Find Best Angle using Projection
    scores = []
    angles = np.arange(-5, 5.1, 0.5) # Steps of 0.5 are faster and usually sufficient

    for angle in angles:
        # Rotate the tiny thresholded image
        (h_s, w_s) = thresh.shape[:2]
        center_s = (w_s // 2, h_s // 2)
        M_s = cv2.getRotationMatrix2D(center_s, angle, 1.0)
        rotated = cv2.warpAffine(thresh, M_s, (w_s, h_s), flags=cv2.INTER_NEAREST)
        
        # Calculate score (Variance of row sums)
        # High variance = clear lines of text (white) vs background (black)
        score = np.var(np.sum(rotated, axis=1))
        scores.append(score)

    best_angle = angles[np.argmax(scores)]
    print(f"      [Deskew] Best angle found: {best_angle:.2f}")

    # 5. Apply to Original Image
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    
    # Use BORDER_CONSTANT with White (255,255,255) to fill corners
    corrected = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    
    # 6. Convert back to PIL (RGB)
    return Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))

    # 7. Save
    df.to_excel(excel_path, index=False)
    print(f"      -> Logged Costs. P1 Total: ${round(ip_cost_p1 + op_cost_p1, 4)}")
