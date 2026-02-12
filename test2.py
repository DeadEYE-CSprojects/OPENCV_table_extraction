final_prompt = (
        "You are an intelligent document digitization system.\n\n"
        "STEP 1: VISUAL ANALYSIS\n"
        "Scan the image. Does it contain significant handwritten text (e.g., filled forms, letters, notes)?\n"
        "- Note: A single signature at the bottom does NOT trigger Handwriting Mode.\n\n"
        "STEP 2: EXECUTION\n"
        "Based on your analysis, strictly apply ONE of the following protocols:\n\n"
        f"{prompt_handwriting}\n\n"
        "OR\n\n"
        f"{prompt_ocr}\n\n"
        "FINAL OUTPUT RULE:\n"
        "Output ONLY the transcribed text. Do not explain which mode you chose."
    )
