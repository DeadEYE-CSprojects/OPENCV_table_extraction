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
