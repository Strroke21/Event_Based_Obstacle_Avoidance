import cv2
import numpy as np

def main():
    # Initialize the mono camera
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Read the first frame and convert it to grayscale
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference between current and previous frame
        diff = cv2.absdiff(prev_gray, gray)

        # Threshold to detect white pixel regions
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Optional: Clean noise
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Convert threshold image to color so we can draw colored rectangles
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # Find contours in threshold (i.e., white pixel regions)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # Adjust to remove noise
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(thresh_colored, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Show result with bounding boxes on white pixel regions
        cv2.imshow('White Pixels with Bounding Boxes', thresh_colored)
        #cv2.imshow('Threshold Binary Image', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update previous frame
        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



