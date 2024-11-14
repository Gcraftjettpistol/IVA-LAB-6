import cv2
import sys

# Load the video
video_path = "C:\\Users\\spooj\\Downloads\\t2.mp4"  # Update with the correct video path
cap = cv2.VideoCapture(video_path)

# Check if video is loaded correctly
if not cap.isOpened():
    print("Error: Couldn't open video.")
    sys.exit()  # Use sys.exit() to exit the program

# Initialize variables
people_count = 0
prev_centroids = []  # Track centroids of the moving objects in the previous frame
entered_centroids = []  # Track centroids that have entered the ROI

# Define the ROI (Region of Interest) for the entrance area (adjust the coordinates as per your video)
roi_x, roi_y, roi_w, roi_h = 200, 100, 400, 200  # Example values, adjust as necessary

# Create a Background Subtractor to detect moving objects
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Loop through video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold the mask to make it binary
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Dilate the thresholded mask to fill in gaps
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_centroids = []  # List to store centroids in the current frame

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Filter out small contours (noise)
            continue

        # Get the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        # Define the center of the bounding box
        center = (x + w // 2, y + h // 2)

        # Add the current centroid to the list
        current_centroids.append(center)

        # Check if the centroid has entered the ROI and not drawn yet
        if center not in entered_centroids and roi_x < center[0] < roi_x + roi_w and roi_y < center[1] < roi_y + roi_h:
            entered_centroids.append(center)
            people_count += 1  # Increment count when entering the ROI

        # Only draw bounding boxes for people who have entered the ROI
        if center in entered_centroids:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the centroid of the person
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Update the previous centroids for the next frame
    prev_centroids = current_centroids

    # Draw the ROI rectangle on the frame
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

    # Display the number of people in the shop
    cv2.putText(frame, f"People in Region of Interest: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("People Counting", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
