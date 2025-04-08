import cv2
import numpy as np

def get_grid_boxes(shape, rows=3, cols=3):
    h, w = shape
    box_h, box_w = h // rows, w // cols
    boxes = []
    for i in range(rows):
        for j in range(cols):
            x1, y1 = j * box_w, i * box_h
            x2, y2 = x1 + box_w, y1 + box_h
            boxes.append((x1, y1, x2, y2))
    return boxes

def distance_to_nearest_white(thresh, box, white_pixels):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if len(white_pixels[0]) == 0:
        return float('inf')

    dists = np.sqrt((white_pixels[1] - cx)**2 + (white_pixels[0] - cy)**2)
    return np.min(dists)

def main():
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        h, w = thresh.shape
        boxes = get_grid_boxes((h, w), 3, 3)

        white_pixels = np.where(thresh == 255)
        empty_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = box
            grid = thresh[y1:y2, x1:x2]
            if cv2.countNonZero(grid) == 0:
                empty_boxes.append(box)

        # Find farthest empty box
        farthest_box = None
        max_dist = -1
        for box in empty_boxes:
            dist = distance_to_nearest_white(thresh, box, white_pixels)
            if dist > max_dist:
                max_dist = dist
                farthest_box = box

        # Convert threshold to color to draw colored boxes
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(thresh_colored, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Light blue grid

        if farthest_box:
            x1, y1, x2, y2 = farthest_box
            cv2.rectangle(thresh_colored, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

        cv2.imshow('Binary Threshold with Grid', thresh_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
