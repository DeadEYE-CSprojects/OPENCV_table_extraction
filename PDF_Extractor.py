import base64
import os
from openai import OpenAI

# ==========================================
# 1. CONFIGURATION
# ==========================================
# ❗ Replace with your actual key
API_KEY = "YOUR_OPENAI_API_KEY" 

# If you are using your corporate gateway, update this:
# BASE_URL = "https://gateway.ai.humana.com/openai/deployments/gpt-4o" 
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL, ...)

# For standard OpenAI usage:
client = OpenAI(api_key=API_KEY)

IMAGE_PATH = "test_document.png" # <--- PUT YOUR IMAGE NAME HERE

# ==========================================
# 2. HELPER: ENCODE IMAGE
# ==========================================
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# 3. THE "STRUCTURE PRESERVING" PROMPT
# ==========================================
def get_ocr_prompt():
    return """
    You are an advanced Optical Character Recognition (OCR) and Document Layout Analysis engine.
    
    YOUR TASK:
    1. Transcribe the text from the provided image EXACTLY as it appears.
    2. PRESERVE THE STRUCTURE: 
       - If there is a table, output a Markdown table.
       - If there are columns, represent them visually or logically grouping them.
       - If there are checkboxes, mark them as [x] or [ ].
    3. OVERLAPPING TEXT: 
       - If a stamp or signature overlaps text, prioritize extracting the underlying typed text first. 
       - Then, note the stamp/signature content in brackets, e.g., "[Stamp: APPROVED]".
    4. NO SUMMARIZATION: Do not explain the document. Do not summarize it. Output ONLY the content.
    
    Output Format: Markdown
    """

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def process_image():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: File {IMAGE_PATH} not found.")
        return

    print(f"Encoding image: {IMAGE_PATH}...")
    base64_image = encode_image(IMAGE_PATH)

    print("Sending to GPT-4o (Detail: HIGH)...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": get_ocr_prompt()
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Extract the structured text from this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                # ❗ CRITICAL: 'high' forces the model to look at 512x512 tiles
                                "detail": "high" 
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096,
            temperature=0  # ❗ CRITICAL: 0 reduces hallucinations
        )

        result = response.choices[0].message.content
        
        print("\n" + "="*40)
        print("GPT-4o EXTRACTION RESULT")
        print("="*40 + "\n")
        print(result)
        
        # Save to file
        with open("gpt4o_output.md", "w", encoding="utf-8") as f:
            f.write(result)
        print("\nSaved output to 'gpt4o_output.md'")

    except Exception as e:
        print(f"Error calling API: {e}")

if __name__ == "__main__":
    process_image()
