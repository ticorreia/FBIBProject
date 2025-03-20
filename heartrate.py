import dlib
import cv2
import numpy as np

# Load face detector and landmark predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        landmarks = dlib_facelandmark(gray, face)

        # Indices for the right cheek
        right_cheek_indices = [16, 35, 53, 54, 55, 12, 13, 14, 15, 16]
        right_cheek = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_cheek_indices], dtype=np.int32)

        # Indices for the left cheek
        left_cheek_indices = [0, 31, 49, 48, 59, 4, 3, 2, 1, 0]  
        left_cheek = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_cheek_indices], dtype=np.int32)

        # Indices for the forehead (upper eyebrow area)
        forehead_indices = [19, 20, 21, 22, 23, 24]
        forehead = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in forehead_indices], dtype=np.int32)

        # Indices for the nose (bridge and tip)
        nose_indices = [27, 28, 29, 30]
        nose = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in nose_indices], dtype=np.int32)

        # Draw the ROIs
        cv2.polylines(frame, [right_cheek], isClosed=True, color=(0, 255, 0), thickness=1)   # Green for right cheek
        cv2.polylines(frame, [left_cheek], isClosed=True, color=(255, 0, 0), thickness=1)    # Blue for left cheek
        cv2.polylines(frame, [forehead], isClosed=False, color=(0, 0, 255), thickness=2)    # Red for forehead
        cv2.polylines(frame, [nose], isClosed=False, color=(255, 255, 0), thickness=2)      # Yellow for nose

    cv2.imshow("Face ROIs: Cheeks, Forehead, Nose", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
