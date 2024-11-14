import cv2
import math

# Load the video
cap = cv2.VideoCapture(r"D:\Assignments\image and vdo\LAB\LAB 6\task4.mp4")

# Check if video is loaded successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Get the video frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the ROI to the lower-bottom part of the window
roi_height = 100  # Height of the ROI from the bottom
roi_x1 = 0
roi_y1 = frame_height - roi_height
roi_x2 = frame_width
roi_y2 = frame_height

# Initialize counters
enter_count = 0
exit_count = 0

# Variables to track the direction of motion
prev_center = None
direction = None

# Set to store unique centers of people who have entered/exited
tracked_people = set()

# Distance threshold to consider people as the same person (in pixels)
distance_threshold = 30

def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours (representing moving objects)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # Filter out small areas
            continue
        
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)
        
        # Define the ROI area and check if the person is inside it
        if x > roi_x1 and y + h > roi_y1 and x + w < roi_x2 and y + h < roi_y2:
            # Get the center of the bounding box
            center = (x + w // 2, y + h // 2)
            
            # Check if this person has already been tracked
            person_found = False
            for tracked in tracked_people:
                if distance(center, tracked) < distance_threshold:
                    person_found = True
                    break
            
            if not person_found:
                # Draw a bounding box around the detected person
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Track direction (left to right or right to left)
                if prev_center:
                    if center[0] > prev_center[0]:
                        direction = 'enter'
                    elif center[0] < prev_center[0]:
                        direction = 'exit'
                
                # Update the previous center point
                prev_center = center
                
                # Count people entering or exiting
                if direction == 'enter':
                    enter_count += 1
                    direction = None  # Reset direction after counting
                elif direction == 'exit':
                    exit_count += 1
                    direction = None  # Reset direction after counting
                
                # Add this person's center to the set to track
                tracked_people.add(center)
    
    # Display the total count on the frame
    cv2.putText(frame, f'Entered: {enter_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Exited: {exit_count}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Draw ROI for debugging (lower-bottom region)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
    
    # Show the processed video frame
    cv2.imshow('Processed Frame', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
