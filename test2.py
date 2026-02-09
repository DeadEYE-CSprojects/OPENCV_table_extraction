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


    # 7. Save
    df.to_excel(excel_path, index=False)
    print(f"      -> Logged Costs. P1 Total: ${round(ip_cost_p1 + op_cost_p1, 4)}")




import numpy as np
import cv2
from PIL import Image

def deskew_image(pil_image):
    """
    ROBUST DESKEW: Handles Major (90/270) and Minor (±5) rotations.
    
    Logic:
    1. Check Orientation: Are words 'wide' (Horizontal) or 'tall' (Vertical)?
    2. Fix 90-degree rotations based on word shape.
    3. Fine-tune the remaining tilt using Projection Profiles (Variance of text lines).
    """
    if pil_image is None: return None
    
    print("\n   --- [Deskew] Starting Analysis ---")
    
    # 1. Convert PIL to OpenCV (BGR)
    img = np.array(pil_image)
    if len(img.shape) == 3:
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h_orig, w_orig = img.shape[:2]
    
    # 2. Create a Binary Map (Text = White, Background = Black)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding automatically finds the best ink separation
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # =========================================================
    # STAGE 1: MACRO CORRECTION (Detect 0 vs 90 degrees)
    # =========================================================
    # We dilate (thicken) text to merge letters into "Word Blocks"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours of these blocks
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    aspect_ratios = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter noise: Box must be reasonable size
        if w > 20 and h > 20: 
            # Ratio > 1 means Tall (Vertical Text). Ratio < 1 means Wide (Horizontal).
            aspect_ratios.append(h / w)
            
    # Calculate the median aspect ratio of all words on the page
    if aspect_ratios:
        median_ratio = np.median(aspect_ratios)
    else:
        median_ratio = 0.5 # Default to horizontal if empty

    rotation_needed = 0
    
    # If Median Ratio > 1.2, words are standing up -> Page is Sideways
    if median_ratio > 1.2:
        print(f"   -> Detected VERTICAL text (Ratio: {median_ratio:.2f}). Rotating 90 deg.")
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        thresh = cv2.rotate(thresh, cv2.ROTATE_90_CLOCKWISE) # Update thresh for Stage 2
        rotation_needed = 90
    else:
        print(f"   -> Detected HORIZONTAL text (Ratio: {median_ratio:.2f}). Keeping orientation.")

    # =========================================================
    # STAGE 2: MICRO CORRECTION (Projection Profile Method)
    # =========================================================
    # Now that we know it's roughly horizontal, we fix the wobble.
    
    print("   -> Calculating optimal projection angle...")
    
    h, w = img.shape[:2]
    scores = []
    # Search range: -5 to +5 degrees
    angles = np.arange(-5, 5.1, 0.5) 
    
    for angle in angles:
        # Rotate ONLY the binary map (Fast)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated_thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Calculate Row Variance (High Variance = Crisp Lines)
        # Sum white pixels across rows
        hist = np.sum(rotated_thresh, axis=1)
        # Calculate how "spiky" the histogram is
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append(score)

    # Find the angle with the sharpest text lines
    best_micro_angle = angles[np.argmax(scores)]
    print(f"   -> Best Fine-Tune Angle: {best_micro_angle:.2f} degrees")
    
    # =========================================================
    # STAGE 3: APPLY FINAL ROTATION
    # =========================================================
    
    # If we need to rotate (either Macro or Micro)
    if rotation_needed != 0 or abs(best_micro_angle) > 0.1:
        # Combined rotation is tricky, simpler to apply fine-tune to the current 'img'
        # (Since 'img' might have already been rotated 90 in Stage 1)
        
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_micro_angle, 1.0)
        
        # Warp with White Border (255,255,255) to prevent black corners
        img = cv2.warpAffine(
            img, 
            M, 
            (w, h), 
            flags=cv2.INTER_CUBIC, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(255, 255, 255)
        )
        print(f"   -> Applied Correction. Total Rotation: {rotation_needed + best_micro_angle:.2f}")
    else:
        print("   -> Image is already straight.")

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))






prompt = (
    "Analyze the visual layout of this document page. "
    "Determine if it is 'Complex' (requiring heavy processing) or 'Simple' (standard text).\n\n"
    "Respond 'NO' (Simple) if the page contains:\n"
    "- Standard paragraphs, articles, or reading text.\n"
    "- Simple bulleted lists, numbered lists, or a Table of Contents.\n"
    "- Basic headers, footers, or page numbers.\n"
    "- A standard letter or memo format.\n"
    "- 2-column text (like a research paper) WITHOUT data tables.\n\n"
    "Respond 'YES' (Complex) ONLY if the page contains:\n"
    "- A dense data table with strict rows/columns and numerical values.\n"
    "- A form with input boxes, checkboxes, or significant handwriting.\n"
    "- A layout with 3+ complex columns (like a newspaper or brochure).\n\n"
    "Answer ONLY with 'YES' or 'NO'."
)
