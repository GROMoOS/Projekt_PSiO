import cv2
import numpy as np
import dlib
import imutils
from math import hypot, atan2, degrees

# Loading Camera and Glasses image and Creating mask
cap = cv2.VideoCapture("videoplayback360p.mp4")
glasses_image = cv2.imread("sunglasses.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
glasses_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while frame is not None:
    glasses_mask.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(frame)

    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Eyes coordinates
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)
        center_eyes = (landmarks.part(27).x, landmarks.part(27).y)
        eyes_lower = (landmarks.part(28).x, landmarks.part(28).y)

        eyes_width = int(hypot(left_eye[0] - right_eye[0],
                               left_eye[1] - right_eye[1]) * 1.5)
        eyes_height = int(hypot(center_eyes[0] - eyes_lower[0],
                                center_eyes[1] - eyes_lower[1]) * 2.2)

        # Adjusting glasses size and rotation
        angle = degrees(atan2(left_eye[0] - right_eye[0], left_eye[1] - right_eye[1]) * -1) - 90

        glasses = cv2.resize(glasses_image, (eyes_width, eyes_height))
        glasses = imutils.rotate_bound(glasses, angle)

        glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)
        _, glasses_mask = cv2.threshold(glasses_gray, 5, 150, cv2.THRESH_BINARY_INV)

        padding = int(eyes_width / 2)
        extended_frame = np.pad(frame, [(padding, padding), (padding, padding), (0, 0)], mode='constant', constant_values=0)

        # New glasses position
        top_left = (int(center_eyes[0] + padding - glasses.shape[1] / 2),
                    int(center_eyes[1] + padding - glasses.shape[0] / 2))

        glasses_area = extended_frame[top_left[1]: top_left[1] + glasses.shape[0],
                                      top_left[0]: top_left[0] + glasses.shape[1]]

        glasses_area_no_glasses = cv2.bitwise_and(glasses_area, glasses_area, mask=glasses_mask)
        final_glasses = cv2.add(glasses_area_no_glasses, glasses)

        extended_frame[top_left[1]: top_left[1] + glasses.shape[0],
                       top_left[0]: top_left[0] + glasses.shape[1]] = final_glasses
        frame = extended_frame[padding: padding + rows,
                            padding: padding + cols]

    cv2.imshow("Frame", frame)

    _, frame = cap.read()

    key = cv2.waitKey(1)
    if key == 27:
        break
# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
