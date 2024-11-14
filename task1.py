import cv2
import numpy as np

# Load the video
cap = cv2.VideoCapture(r"D:\Assignments\image and vdo\LAB\LAB 6\task1-nig.mp4")

# using the pre-trained Haar Cascade for human detection (using full body detection)
#download https://github.com/opencv/opencv/tree/master/data/haarcascades
human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "D:\Assignments\image and vdo\LAB\LAB 6\haarcascade_fullbody.xml")

# Define the skin tone filter for black skin (HSV range)
skin_lower = np.array([0, 20, 70], dtype=np.uint8)  # Lower bound for skin color (HSV)
skin_upper = np.array([20, 150, 255], dtype=np.uint8)  # Upper bound for skin color (HSV)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for Haar cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect humans (full body) using Haar cascade
    humans = human_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in humans:
        # Extract the region of interest (ROI) for skin detection
        roi = frame[y:y+h, x:x+w]
        
        # Convert the ROI to HSV color space for skin tone detection
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Apply the skin tone filter
        skin_mask = cv2.inRange(hsv_roi, skin_lower, skin_upper)
        
        # Check if the skin mask has a significant number of white pixels
        if np.sum(skin_mask) > 5000:  # This threshold might need adjustments
            # Person is likely a black person, so mark with a green dot
            center = (x + w // 2, y + h // 2)  # Use center of the bounding box
            
            # Draw a green dot at the center of the detected person
            cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Solid green dot
            cv2.putText(frame, "Black Person", (center[0] - 40, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video Tracking', frame)

    # Press 'q' to quit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
