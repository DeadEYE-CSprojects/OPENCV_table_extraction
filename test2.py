import threading
import time
from google.genai import types
from PIL import Image

def llm_convert_to_text(pil_image, filename="Unknown", page_num="Unknown"):
    """
    Threaded OCR with Timeout AND Token Counting.
    """
    
    # --- CONFIGURATION ---
    TIMEOUT_SEC = 600       # 10 Minutes
    RETRY_DELAY = 150       # 2.5 Minutes
    MAX_ATTEMPTS = 2        # Run once + 1 Retry
    
    # 1. Optimize Image
    optimized_img = pil_image.copy()
    max_dimension = 2048
    if max(optimized_img.size) > max_dimension:
        optimized_img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    # 2. Prompt
    prompt = (
        "You are a robotic OCR engine. Copy pixels to text exactly.\n"
        "RULES:\n"
        "1. Start response with text and end with text.\n"
        "2. Preserve spatial layout using spaces.\n"
        "3. Draw tables using Markdown pipes (|).\n"
        "4. No summarization."
    )

    # --- HELPER: WORKER FUNCTION ---
    # We use a mutable dict 'result_container' to get data out of the thread
    def api_worker(result_container):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-pro-exp",
                contents=[prompt, optimized_img],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192
                )
            )
            result_container['response'] = response
        except Exception as e:
            result_container['error'] = e

    # --- MAIN LOOP ---
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"      -> [LLM] Attempt {attempt} (Timeout: {TIMEOUT_SEC}s)...")
        
        result = {}
        t = threading.Thread(target=api_worker, args=(result,))
        t.start()
        
        # Wait for thread to finish
        t.join(timeout=TIMEOUT_SEC)

        # Initialize defaults
        in_tokens = 0
        out_tokens = 0
        error_msg = "Unknown Error"

        if t.is_alive():
            print(f"      [Warning] TIMEOUT: Page stuck for >{TIMEOUT_SEC}s.")
            error_msg = "Timeout"
        
        elif 'error' in result:
            print(f"      [Warning] API Error: {result['error']}")
            error_msg = str(result['error'])
            
        elif 'response' in result:
            # SUCCESS CASE
            try:
                resp = result['response']
                text = resp.text
                
                # --- EXTRACT TOKENS HERE ---
                if hasattr(resp, 'usage_metadata') and resp.usage_metadata:
                    in_tokens = resp.usage_metadata.prompt_token_count
                    out_tokens = resp.usage_metadata.candidates_token_count
                
                if text:
                    return text, in_tokens, out_tokens
                    
            except Exception as e:
                print(f"      [Error] Parsing response failed: {e}")
                error_msg = "Parse Error"

        # --- RETRY LOGIC ---
        if attempt < MAX_ATTEMPTS:
            print(f"      -> [Wait] Taking a break for {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
        else:
            # FINAL FAILURE
            print(f"      [Fail] Skipping {filename} Page {page_num}.")
            try:
                with open("process_log.txt", "a") as logf:
                    logf.write(f"{filename}: {page_num} -- > skipped (Reason: {error_msg})\n")
            except: pass
            
            return "", 0, 0

    return "", 0, 0
