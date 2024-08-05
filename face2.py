import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)


def draw_text_with_border(image, text, position, font, scale, text_color, border_color, thickness, border_thickness):
    # Gambar border
    cv2.putText(image, text, position, font, scale, border_color, border_thickness, cv2.LINE_AA)
    # Gambar teks
    cv2.putText(image, text, position, font, scale, text_color, thickness, cv2.LINE_AA)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera")
        break

    # Konversi frame ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    results = face_detection.process(rgb_frame)

    # Periksa apakah ada deteksi wajah
    if results.detections:
        text = "Ada manusia"
        text_color = (0, 255, 0)  # Warna hijau untuk teks
    else:
        text = "Tidak terdeteksi manusia"
        text_color = (0, 0, 255)  # Warna merah untuk teks

    border_color = (0, 0, 0)  # Warna hitam untuk border

    # Tampilkan teks di frame dengan border
    draw_text_with_border(frame, text, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, border_color, 2, 4)

    # Tampilkan frame (tanpa flip/mirror)
    cv2.imshow('Deteksi Manusia', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
