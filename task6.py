import cv2
import numpy as np

# Load the video
video_path = "D:\\Assignments\\image and vdo\\LAB\\LAB 6\\task6 2.mp4"  # Your video path
cap = cv2.VideoCapture(video_path)

# Check if the video opened correctly
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Variables for counting red cars and tracking unique detections
red_car_count = 0
frame_count = 0
detected_bboxes = []  # List to store bounding boxes of detected red cars

# Red color range in HSV space
lower_red1 = np.array([0, 120, 70])  # Lower bound of red color
upper_red1 = np.array([10, 255, 255])  # Upper bound of red color
lower_red2 = np.array([170, 120, 70])  # Another red range
upper_red2 = np.array([180, 255, 255])  # Another red range

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop when video ends
    
    frame_count += 1

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for red color (both ranges)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours to find potential red cars
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Minimum area to filter out noise
            # Get bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            car_bbox = (x, y, w, h)

            # Check if this bounding box has already been detected (within a threshold distance)
            car_detected = False
            for detected_bbox in detected_bboxes:
                # Calculate the center of the bounding box
                cx, cy = x + w // 2, y + h // 2
                detected_cx, detected_cy = detected_bbox[0] + detected_bbox[2] // 2, detected_bbox[1] + detected_bbox[3] // 2
                
                # Calculate Euclidean distance between centers to avoid counting the same car
                distance = np.linalg.norm(np.array([cx, cy]) - np.array([detected_cx, detected_cy]))
                
                if distance < 100:  # If the distance between centers is small, it's the same car
                    car_detected = True
                    break

            # If this car has not been detected yet, count it and add its bounding box to the list
            if not car_detected:
                detected_bboxes.append(car_bbox)
                red_car_count += 1

            # Draw the bounding box around the red car
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for detection

    # Display the number of red cars detected on top of the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Red Cars: {red_car_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with detection
    cv2.imshow('Red Car Detection', frame)

    # Stop if 'q' is pressed or if we reach the count of 4 cars
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Output the final count of red cars
print(f"Total number of red cars detected: {red_car_count}")
