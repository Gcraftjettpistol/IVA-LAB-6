import cv2
import numpy as np

# Load the reference (fraud) image
reference_image_path = r"D:/Assignments/image and vdo/LAB/LAB 6/fraud.png"  # Path to the reference image (fraud)
reference_img = cv2.imread(reference_image_path)
if reference_img is None:
    print(f"Error loading reference image: {reference_image_path}")
    exit()
gray_reference = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect face in the reference image
reference_faces = face_cascade.detectMultiScale(gray_reference, scaleFactor=1.1, minNeighbors=5)
if len(reference_faces) == 0:
    print("No face detected in the reference image.")
    exit()
else:
    # Use the first detected face
    x, y, w, h = reference_faces[0]
    reference_face = gray_reference[y:y+h, x:x+w]

# Load the target image to compare with the reference image
target_image_path = r"C:\Users\spooj\Downloads\task3.jpg"  # Path to the target image
target_img = cv2.imread(target_image_path)
if target_img is None:
    print(f"Error loading target image: {target_image_path}")
    exit()
gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# Detect face in the target image
target_faces = face_cascade.detectMultiScale(gray_target, scaleFactor=1.1, minNeighbors=5)
if len(target_faces) == 0:
    print("No face detected in the target image.")
    exit()
else:
    # Use the first detected face
    x, y, w, h = target_faces[0]
    target_face = gray_target[y:y+h, x:x+w]

# Resize faces to a common size for comparison (100x100)
resized_reference_face = cv2.resize(reference_face, (100, 100))
resized_target_face = cv2.resize(target_face, (100, 100))

# Compute the similarity score using Mean Squared Error (MSE)
mse = np.sum((resized_reference_face - resized_target_face) ** 2) / (100 * 100)

# Set a threshold based on experimentation; lower MSE means better match
threshold = 1000  # Example threshold value, adjust based on results
if mse >= threshold:
    print(f"Faces did not match. MSE: {mse}")  # Print MSE and say faces did not match

# Optionally, visualize the faces and results
cv2.rectangle(reference_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(reference_img, "Reference Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.rectangle(target_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(target_img, "Target Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show the images with rectangles drawn around faces
cv2.imshow("Reference Image", reference_img)
cv2.imshow("Target Image", target_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
