import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk, ImageDraw, ImageFont

class HumanDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection")
        self.root.geometry("800x600")

        # Inisialisasi MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection()

        # Buka kamera (0 untuk kamera default)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            self.root.quit()
            return

        # Label untuk menampilkan video
        self.video_label = Label(root)
        self.video_label.pack(fill="both", expand=True)

        # Inisialisasi variabel untuk deteksi manusia
        self.human_detected = False

        # Mulai video loop
        self.update_video()

    def update_video(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Konversi frame ke RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Deteksi wajah
                results = self.face_detection.process(frame_rgb)

                # Jika ada wajah yang terdeteksi
                if results.detections:
                    self.human_detected = True
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                            int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                else:
                    self.human_detected = False

                # Konversi frame ke format RGB untuk PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                # Tambahkan teks peringatan
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()
                if self.human_detected:
                    draw.text((10, 10), "Ada manusia", font=font, fill=(0, 255, 0))
                else:
                    draw.text((10, 10), "Tidak terdeteksi manusia", font=font, fill=(255, 0, 0))

                # Konversi frame RGB ke ImageTk
                image_tk = ImageTk.PhotoImage(image)

                # Perbarui label dengan gambar baru
                self.video_label.img_tk = image_tk
                self.video_label.configure(image=image_tk)

            # Panggil metode ini lagi setelah 30 ms
            self.root.after(30, self.update_video)
        except Exception as e:
            print(f"Error updating video: {e}")

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

# Buat jendela Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = HumanDetectorApp(root)
    root.mainloop()
