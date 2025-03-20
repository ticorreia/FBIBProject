import dlib
import cv2
import numpy as np

# Load face detector and landmark predictor
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

video_path = r"C:\Users\ticor\Desktop\LEBiol22\Fundamentos de Biossinais e Imagiologia Biomédica\MatLabFBIB\Projeto\subject_1.avi"
cap = cv2.VideoCapture(video_path)

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

    # Create a black mask for ROI extraction
    roi_mask = np.zeros_like(frame)

    for face in faces:
        landmarks = dlib_facelandmark(gray, face)

        # Right Cheek ROI
        right_cheek_indices = [16, 46, 47, 35, 53, 54, 55, 12, 13, 14, 15, 16]
        right_cheek = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in right_cheek_indices], dtype=np.int32)

        # Left Cheek ROI
        left_cheek_indices = [0, 41, 40, 31, 49, 48, 59, 4, 3, 2, 1, 0]
        left_cheek = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in left_cheek_indices], dtype=np.int32)

        # Forehead ROI (Higher!)
        forehead_indices = [19, 20, 21, 22, 23, 24]  # Eyebrow area
        forehead_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in forehead_indices], dtype=np.int32)

        # Extend forehead upwards dynamically
        eyebrow_mid_y = (landmarks.part(19).y + landmarks.part(24).y) // 2
        nose_bridge_y = landmarks.part(27).y
        forehead_extension = (nose_bridge_y - eyebrow_mid_y) * 1.5
        forehead_extended = np.array([(x, y - int(forehead_extension)) for (x, y) in forehead_points], dtype=np.int32)
        forehead_full = np.vstack((forehead_points, forehead_extended))
        forehead_hull = cv2.convexHull(forehead_full)

        # Nose ROI (Convex Hull)
        nose_indices = [27, 39, 40, 31, 32, 33, 34, 35, 47, 42]
        nose_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in nose_indices], dtype=np.int32)
        nose_hull = cv2.convexHull(nose_points)

        # 🖍 Draw the ROIs on the original frame
        cv2.polylines(frame, [right_cheek], isClosed=True, color=(0, 255, 0), thickness=2)  # Green - Right Cheek
        cv2.polylines(frame, [left_cheek], isClosed=True, color=(255, 0, 0), thickness=2)   # Blue - Left Cheek
        cv2.polylines(frame, [forehead_hull], isClosed=True, color=(0, 0, 255), thickness=2)  # Red - Forehead
        cv2.polylines(frame, [nose_hull], isClosed=True, color=(255, 255, 0), thickness=2)  # Yellow - Nose

        # 🖍 Fill the ROIs with the original image in the mask
        cv2.fillPoly(roi_mask, [right_cheek], (255, 255, 255))
        cv2.fillPoly(roi_mask, [left_cheek], (255, 255, 255))
        cv2.fillPoly(roi_mask, [forehead_hull], (255, 255, 255))
        cv2.fillPoly(roi_mask, [nose_hull], (255, 255, 255))

    # Apply the mask to keep only ROI areas from the original frame
    extracted_rois = cv2.bitwise_and(frame, roi_mask)

    # Show the full face with ROIs
    cv2.imshow("Face ROIs: Cheeks, Forehead, Nose", frame)

    # Show the extracted ROIs
    cv2.imshow("Extracted ROIs (Masked)", extracted_rois)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
