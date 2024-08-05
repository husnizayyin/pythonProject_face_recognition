import cv2
import mediapipe as mp
import time
import simpleaudio as sa

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the warning sound
wave_obj = sa.WaveObject.from_wave_file("alert.wav")
play_obj = None

# Initialize webcam
cap = cv2.VideoCapture(0)

person_present = True  # Assume a person is present initially
last_seen_time = time.time()
alert_threshold = 5  # Seconds of inactivity before triggering an alert

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(image)

    if results.pose_landmarks:
        person_present = True
        last_seen_time = time.time()
        if play_obj is not None:
            play_obj.stop()
            play_obj = None
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        person_present = False

    # Check for inactivity
    if not person_present:
        if (time.time() - last_seen_time) > alert_threshold:
            print("Alert: No person detected in front of the laptop!")
            if play_obj is None:
                play_obj = wave_obj.play()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()