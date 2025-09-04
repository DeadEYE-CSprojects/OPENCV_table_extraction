# OPENCV_table_extraction

def process_table(img, table_box, page_num, table_num):
    """
    Crops each cell from a table using a robust method that reconstructs the
    grid from line intersections, correctly handling merged and irregular cells.
    """
    tx, ty, tw, th = table_box
    table_img = img[ty:ty+th, tx:tx+tw]
    
    cell_output_dir = os.path.join(TEMP_FOLDER, "cell_images", f"Page_{page_num}_Table_{table_num}")
    os.makedirs(cell_output_dir, exist_ok=True)

    table_gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
    table_binary = cv2.adaptiveThreshold(~table_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)
    
    # 1. Isolate the horizontal and vertical lines
    cell_hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CELL_HOR_KERNEL_SIZE)
    detected_hor = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, cell_hor_kernel, iterations=2)
    
    cell_ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CELL_VER_KERNEL_SIZE)
    detected_ver = cv2.morphologyEx(table_binary, cv2.MORPH_OPEN, cell_ver_kernel, iterations=2)

    # 2. Find the intersection points of the lines
    intersections = cv2.bitwise_and(detected_hor, detected_ver)
    
    # Save debug images
    cv2.imwrite(os.path.join(cell_output_dir, f"__debug_01_horizontal_lines.png"), detected_hor)
    cv2.imwrite(os.path.join(cell_output_dir, f"__debug_02_vertical_lines.png"), detected_ver)
    cv2.imwrite(os.path.join(cell_output_dir, f"__debug_03_intersections.png"), intersections)

    # 3. Get coordinates of intersection points
    intersection_contours, _ = cv2.findContours(intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    intersection_points = []
    for contour in intersection_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            intersection_points.append((cX, cY))

    if len(intersection_points) < 4:
        warning_msg = f"  Warning: Not enough intersection points ({len(intersection_points)}) to form a grid in Table {table_num} on Page {page_num}."
        print(warning_msg)
        return "Could not process table due to lack of clear grid intersections."

    # 4. Get unique X and Y coordinates from the intersection points to form the grid map
    unique_x = sorted(list(set(p[0] for p in intersection_points)))
    unique_y = sorted(list(set(p[1] for p in intersection_points)))

    def filter_close_coords(coords, threshold=5):
        if not coords: return []
        filtered = [coords[0]]
        for i in range(1, len(coords)):
            if coords[i] - filtered[-1] > threshold:
                filtered.append(coords[i])
        return filtered

    grid_x = filter_close_coords(unique_x)
    grid_y = filter_close_coords(unique_y)

    # 5. Form cell boxes from the grid of unique coordinates
    raw_cell_boxes = []
    for i in range(len(grid_y) - 1):
        for j in range(len(grid_x) - 1):
            y1, y2 = grid_y[i], grid_y[i+1]
            x1, x2 = grid_x[j], grid_x[j+1]
            w, h = x2 - x1, y2 - y1
            
            # Check if a potential cell is valid by ensuring its center is not a black line
            cell_center_x, cell_center_y = x1 + w // 2, y1 + h // 2
            if w > 5 and h > 5 and table_binary[cell_center_y, cell_center_x] == 255:
                raw_cell_boxes.append((x1, y1, w, h))

    # Visual debug step to show you exactly what cells were found
    table_with_cells_drawn = table_img.copy()
    for x, y, w, h in raw_cell_boxes:
        cv2.rectangle(table_with_cells_drawn, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(cell_output_dir, f"__debug_04_final_cells.png"), table_with_cells_drawn)

    if not raw_cell_boxes:
        warning_msg = f"  Warning: Could not reconstruct cells from the grid map in Table {table_num} on Page {page_num}."
        print(warning_msg)
        return warning_msg

    # Sort by y-coordinate first to group into rows
    raw_cell_boxes.sort(key=lambda box: box[1])
    
    rows, current_row, last_cell_y = [], [], raw_cell_boxes[0][1]

    for box in raw_cell_boxes:
        if abs(box[1] - last_cell_y) > ROW_Y_TOLERANCE and current_row:
            rows.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [box]
            last_cell_y = box[1]
        else:
            current_row.append(box)
    if current_row:
        rows.append(sorted(current_row, key=lambda b: b[0]))
    
    table_content_text = ""
    for r_idx, row in enumerate(rows):
        row_texts = []
        for c_idx, (cx, cy, cw, ch) in enumerate(row):
            # Add a small buffer to avoid cutting off text at the edges
            cx_b, cy_b = cx + 1, cy + 1
            cw_b, ch_b = cw - 2, ch - 2
            
            cell_img = table_img[cy_b:cy_b+ch_b, cx_b:cx_b+cw_b]
            if cell_img.size == 0: continue

            filename = f"page{page_num}_T{table_num}R{r_idx+1}C{c_idx+1}.png"
            cv2.imwrite(os.path.join(cell_output_dir, filename), cell_img)
            
            cell_text = pytesseract.image_to_string(cell_img, config=TESSERACT_CELL_CONFIG).strip()
            row_texts.append(cell_text.replace('\n', ' '))
            
        table_content_text += " | ".join(row_texts) + "\n"
            
    print(f"  Processed Table {table_num} on Page {page_num}, saved {len(raw_cell_boxes)} cell images.")
    return table_content_text


