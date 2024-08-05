import cv2
import mediapipe as mp
import time
import simpleaudio as sa

# Initialize face detection model
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load the warning sound
wave_obj = sa.WaveObject.from_wave_file("alert.wav")
play_obj = None

# Initialize webcam
cap = cv2.VideoCapture(0)

face_present = True  # Assume a face is present initially
last_seen_time = time.time()
alert_threshold = 2  # Seconds of inactivity before triggering an alert

while True:  # Keep running indefinitely
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(image)

    if results.detections:
        face_present = True
        last_seen_time = time.time()
        if play_obj is not None:
            play_obj.stop()
            play_obj = None

    else:
        face_present = False

    if not face_present and (time.time() - last_seen_time) > alert_threshold:
        print("Peringatan: Tidak ada wajah di depan laptop!")
        if play_obj is None:
            play_obj = wave_obj.play()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Face Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()