import cv2
import numpy as np

noise_pixels = 500
merge_distance = 50  # Distance threshold for merging close bounding boxes

def group_rectangles(rectangles, threshold):
    grouped = []
    used = [False] * len(rectangles)

    for i, rect1 in enumerate(rectangles):
        if used[i]:
            continue
        x1, y1, w1, h1 = rect1
        x2, y2 = x1 + w1, y1 + h1
        merged = rect1
        for j, rect2 in enumerate(rectangles):
            if i != j and not used[j]:
                a1, b1, w2, h2 = rect2
                a2, b2 = a1 + w2, b1 + h2

                # Check if they are close enough to merge
                if not (x2 + threshold < a1 or a2 + threshold < x1 or
                        y2 + threshold < b1 or b2 + threshold < y1):
                    # Merge into a larger box
                    x_min = min(x1, a1)
                    y_min = min(y1, b1)
                    x_max = max(x2, a2)
                    y_max = max(y2, b2)
                    merged = (x_min, y_min, x_max - x_min, y_max - y_min)
                    used[j] = True
        grouped.append(merged)
        used[i] = True
    return grouped

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

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Collect bounding boxes of significant contours
        rects = []
        for cnt in contours:
            if cv2.contourArea(cnt) > noise_pixels:
                rects.append(cv2.boundingRect(cnt))

        # Group nearby rectangles
        grouped_rects = group_rectangles(rects, merge_distance)

        # Draw merged/grouped bounding boxes
        for x, y, w, h in grouped_rects:
            cv2.rectangle(thresh_colored, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow('White Pixels with Grouped Bounding Boxes', thresh_colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
