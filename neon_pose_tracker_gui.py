import sys
import cv2
import time
import numpy as np
import mediapipe as mp
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QComboBox, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPalette
from PyQt5.QtCore import QTimer, Qt
from threading import Thread
import imageio
import pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import csv
import pyttsx3
import threading
import webbrowser

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

class Live3DPlot:
    def __init__(self):
        self.coords = []
        self.running = True
        self.paused = False
        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def update(self, coords):
        if not self.paused:
            self.coords = coords

    def run(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.ion()
        while self.running:
            if self.coords and not self.paused:
                ax.cla()
                x, y, z = zip(*self.coords)
                ax.scatter(x, y, z, c='cyan', s=40)
                ax.plot(x, y, z, color='magenta')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_zlim(-0.5, 0.5)
                ax.view_init(elev=10., azim=135)
                fig.canvas.draw()
                fig.canvas.flush_events()
            time.sleep(0.05)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def clear(self):
        self.coords = []

class NeonPoseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neon Pose Tracker")
        self.setGeometry(100, 100, 800, 600)

        self.video_label = QLabel(self)
        self.record_btn = QPushButton("Start Recording")
        self.gif_btn = QPushButton("Export GIF")
        self.music_btn = QPushButton("Play Music")
        self.screenshot_btn = QPushButton("Screenshot")  # Feature 1
        self.export_csv_btn = QPushButton("Export CSV")  # Feature 3
        self.theme_btn = QPushButton("Toggle Theme")     # Feature 6

        # Camera selection (Feature 4)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1"])
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        # 3D Plot Controls (Feature 5)
        self.plot_pause_btn = QPushButton("Pause 3D Plot")
        self.plot_resume_btn = QPushButton("Resume 3D Plot")
        self.plot_clear_btn = QPushButton("Clear 3D Plot")

        # Real-time feedback label (Feature 8)
        self.feedback_label = QLabel("")
        self.feedback_label.setAlignment(Qt.AlignCenter)
        self.feedback_label.setStyleSheet("font-size: 18px; color: green;")

        # 2. Repetition/Exercise Counter
        self.rep_count = 0
        self.rep_label = QLabel("Reps: 0")
        self.rep_label.setAlignment(Qt.AlignCenter)
        self.rep_label.setStyleSheet("font-size: 18px; color: blue;")

        # 3. Custom Pose Alerts
        self.custom_pose_btn = QPushButton("Set Custom Pose Alert")
        self.custom_pose_btn.clicked.connect(self.set_custom_pose)

        # 4. Live Streaming Integration (OBS Virtual Camera hint)
        self.obs_hint_btn = QPushButton("How to Stream Live (OBS)")
        self.obs_hint_btn.clicked.connect(self.show_obs_hint)

        # 8. Leaderboard & Social Sharing
        self.share_btn = QPushButton("Share Session")
        self.share_btn.clicked.connect(self.share_session)

        # Layouts
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.record_btn)
        btn_layout.addWidget(self.gif_btn)
        btn_layout.addWidget(self.music_btn)
        btn_layout.addWidget(self.screenshot_btn)
        btn_layout.addWidget(self.export_csv_btn)
        btn_layout.addWidget(self.theme_btn)

        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot_pause_btn)
        plot_layout.addWidget(self.plot_resume_btn)
        plot_layout.addWidget(self.plot_clear_btn)
        plot_layout.addWidget(self.camera_combo)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(btn_layout)
        layout.addLayout(plot_layout)
        layout.addWidget(self.feedback_label)
        layout.addWidget(self.rep_label)
        layout.addWidget(self.custom_pose_btn)
        layout.addWidget(self.obs_hint_btn)
        layout.addWidget(self.share_btn)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.pose = mp_pose.Pose()
        self.hands = mp_hands.Hands()
        self.recording = False
        self.frames = []
        self.video_writer = None
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_file = 'recorded_session.mp4'
        self.landmark_data = []  # For CSV export

        pygame.mixer.init()
        self.music_playing = False
        self.music_path = 'background.mp3'  # ensure this file exists

        self.plot3d = Live3DPlot()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.record_btn.clicked.connect(self.toggle_recording)
        self.gif_btn.clicked.connect(self.export_gif)
        self.music_btn.clicked.connect(self.toggle_music)
        self.screenshot_btn.clicked.connect(self.save_screenshot)  # Feature 1
        self.export_csv_btn.clicked.connect(self.export_csv)        # Feature 3
        self.theme_btn.clicked.connect(self.toggle_theme)           # Feature 6

        self.plot_pause_btn.clicked.connect(self.plot3d.pause)      # Feature 5
        self.plot_resume_btn.clicked.connect(self.plot3d.resume)    # Feature 5
        self.plot_clear_btn.clicked.connect(self.plot3d.clear)      # Feature 5

        self.fps_time = 0
        self.current_frame = None
        self.dark_theme = False

        self.narrator_engine = pyttsx3.init()
        self.last_narration = ""
        self.narrate_enabled = True  # Toggle for narration
        # Add a button to toggle narration if you want:
        # self.narrate_btn = QPushButton("Toggle Narration")
        # self.narrate_btn.clicked.connect(self.toggle_narration)
        # layout.addWidget(self.narrate_btn)

        # Yoga Flow
        self.yoga_flow = [
            {"name": "T-Pose", "duration": 8, "instruction": "Stand straight with both arms stretched out horizontally."},
            {"name": "Hands Up", "duration": 8, "instruction": "Raise both hands above your head and stand straight."},
        ]
        self.current_pose_idx = 0
        self.pose_hold_time = 0
        self.in_pose = False
        self.yoga_timer = QTimer()
        self.yoga_timer.timeout.connect(self.next_yoga_pose)

        # Add Yoga Flow button
        self.yoga_btn = QPushButton("Start Yoga Flow")
        self.yoga_btn.clicked.connect(self.start_yoga_flow)
        self.layout().addWidget(self.yoga_btn)

        self.custom_pose_landmarks = None

    def get_camera_count(self):
        count = 0
        for i in range(2):  # Only check 0 and 1
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                count += 1
                cap.release()
        return count if count > 0 else 1

    def change_camera(self, idx):
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def toggle_recording(self):
        self.recording = not self.recording
        if self.recording:
            self.video_writer = cv2.VideoWriter(self.output_file, self.fourcc, 20.0, (640, 480))
            self.record_btn.setText("Stop Recording")
        else:
            self.video_writer.release()
            self.video_writer = None
            self.record_btn.setText("Start Recording")

    def toggle_music(self):
        if self.music_playing:
            pygame.mixer.music.stop()
            self.music_btn.setText("Play Music")
        else:
            try:
                pygame.mixer.music.load(self.music_path)
                pygame.mixer.music.play(-1)
                self.music_btn.setText("Stop Music")
            except Exception as e:
                print("Music Error:", e)
        self.music_playing = not self.music_playing

    def export_gif(self):
        if not self.frames:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "pose.gif", "GIF Files (*.gif)")
        if path:
            imageio.mimsave(path, self.frames, duration=0.1)
            self.frames.clear()

    def save_screenshot(self):
        if self.current_frame is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Screenshot", "screenshot.png", "PNG Files (*.png)")
            if path:
                cv2.imwrite(path, self.current_frame)
    # Feature 1

    def export_csv(self):
        if not self.landmark_data:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "landmarks.csv", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['frame', 'landmark_id', 'x', 'y', 'z'])
                for row in self.landmark_data:
                    writer.writerow(row)
            self.landmark_data.clear()
    # Feature 3

    def toggle_theme(self):
        if self.dark_theme:
            self.setStyleSheet("")
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("white"))
            palette.setColor(QPalette.WindowText, QColor("black"))
            self.setPalette(palette)
            self.dark_theme = False
        else:
            self.setStyleSheet("background-color: #222; color: #eee;")
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#222"))
            palette.setColor(QPalette.WindowText, QColor("#eee"))
            self.setPalette(palette)
            self.dark_theme = True
    # Feature 6

    def narrate(self, text):
        if self.narrate_enabled and text and text != self.last_narration:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()
            self.last_narration = text

    def _speak(self, text):
        self.narrator_engine.say(text)
        self.narrator_engine.runAndWait()

    # Optional: to toggle narration on/off
    # def toggle_narration(self):
    #     self.narrate_enabled = not self.narrate_enabled

    def start_yoga_flow(self):
        self.current_pose_idx = 0
        self.pose_hold_time = 0
        self.in_pose = False
        self.yoga_btn.setEnabled(False)
        self.narrate(f"Let's begin! First pose: {self.yoga_flow[0]['name']}. {self.yoga_flow[0]['instruction']}")
        self.feedback_label.setText(f"Yoga Flow: {self.yoga_flow[0]['name']}")
        self.yoga_timer.start(self.yoga_flow[0]['duration'] * 1000)

    def next_yoga_pose(self):
        self.current_pose_idx += 1
        if self.current_pose_idx >= len(self.yoga_flow):
            self.narrate("Yoga flow complete. Great job!")
            self.feedback_label.setText("Yoga flow complete!")
            self.yoga_btn.setEnabled(True)
            self.yoga_timer.stop()
            return
        pose = self.yoga_flow[self.current_pose_idx]
        self.narrate(f"Next pose: {pose['name']}. {pose['instruction']}")
        self.feedback_label.setText(f"Yoga Flow: {pose['name']}")
        self.yoga_timer.start(pose['duration'] * 1000)

    def set_custom_pose(self):
        # Save current pose as custom alert pose
        if hasattr(self, 'last_pose_landmarks') and self.last_pose_landmarks is not None:
            self.custom_pose_landmarks = [(lm.x, lm.y, lm.z) for lm in self.last_pose_landmarks]
            self.narrate("Custom pose alert set.")
            self.feedback_label.setText("Custom pose alert set!")
        else:
            self.feedback_label.setText("No pose detected to set as custom alert.")

    def show_obs_hint(self):
        msg = (
            "To stream your pose-tracked video live:\n"
            "1. Install OBS Studio and the OBS Virtual Camera plugin.\n"
            "2. Add a Window Capture or Screen Capture source for this app in OBS.\n"
            "3. Start the OBS Virtual Camera.\n"
            "4. Select 'OBS Virtual Camera' as your webcam in Zoom, Teams, or browser.\n"
            "Learn more: https://obsproject.com/"
        )
        webbrowser.open("https://obsproject.com/")
        self.feedback_label.setText("Opened OBS instructions in your browser.")

    def share_session(self):
        # For demo: open Twitter share with a message
        url = "https://twitter.com/intent/tweet?text=I+just+completed+a+yoga+session+with+Neon+Pose+Tracker!+%23YogaAI"
        webbrowser.open(url)
        self.feedback_label.setText("Share your session on social media!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)

        h, w, _ = frame.shape
        coords_for_3d = []
        landmarks_this_frame = []

        feedback = ""
        person_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]
        person_id = 0
        pose_correct = False

        # Save last pose landmarks for custom pose alert
        self.last_pose_landmarks = pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None

        if pose_results.pose_landmarks:
            color = person_colors[person_id % len(person_colors)]
            for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z
                coords_for_3d.append((lm.x, lm.y, cz))
                cv2.circle(frame, (cx, cy), 5, color, -1)
                landmarks_this_frame.append([len(self.landmark_data), idx, lm.x, lm.y, cz])
            skeleton_pairs = [
                (11,13), (13,15), (12,14), (14,16),
                (11,12), (23,24),
                (23,25), (25,27), (24,26), (26,28),
                (27,31), (28,32)
            ]
            for a, b in skeleton_pairs:
                pt1 = pose_results.pose_landmarks.landmark[a]
                pt2 = pose_results.pose_landmarks.landmark[b]
                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)

            left_shoulder = pose_results.pose_landmarks.landmark[11]
            right_shoulder = pose_results.pose_landmarks.landmark[12]
            left_wrist = pose_results.pose_landmarks.landmark[15]
            right_wrist = pose_results.pose_landmarks.landmark[16]
            left_hip = pose_results.pose_landmarks.landmark[23]
            right_hip = pose_results.pose_landmarks.landmark[24]

            # 2. Repetition/Exercise Counter (example: count "hands up" reps)
            if not hasattr(self, 'last_hands_up'):
                self.last_hands_up = False
            hands_up = left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y
            if hands_up and not self.last_hands_up:
                self.rep_count += 1
                self.rep_label.setText(f"Reps: {self.rep_count}")
            self.last_hands_up = hands_up

            # 3. Custom Pose Alerts (simple Euclidean distance check)
            if self.custom_pose_landmarks is not None:
                dist = np.mean([
                    np.linalg.norm(np.array([lm.x, lm.y, lm.z]) - np.array(ref))
                    for lm, ref in zip(pose_results.pose_landmarks.landmark, self.custom_pose_landmarks)
                ])
                if dist < 0.08:
                    feedback += "Custom pose matched! "
                    self.narrate("Custom pose matched!")

            # 9. Posture Reminder (example: slouching detection)
            if abs(left_shoulder.y - right_shoulder.y) > 0.15:
                self.bad_posture_frames += 1
                if self.bad_posture_frames > self.posture_reminder_threshold:
                    feedback += "Please correct your posture!"
                    self.narrate("Please correct your posture!")
                    self.bad_posture_frames = 0
            else:
                self.bad_posture_frames = 0

            # --- Yoga Narrator Feedback & Pose Checking ---
            if self.yoga_timer.isActive():
                pose_name = self.yoga_flow[self.current_pose_idx]['name']
                if pose_name == "T-Pose":
                    # Arms stretched horizontally: wrists and shoulders at similar y
                    if abs(left_wrist.y - left_shoulder.y) < 0.07 and abs(right_wrist.y - right_shoulder.y) < 0.07:
                        feedback += "Good! Hold the T-Pose."
                        pose_correct = True
                    else:
                        feedback += "Stretch both arms out horizontally."
                elif pose_name == "Hands Up":
                    if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                        feedback += "Great! Hold hands up."
                        pose_correct = True
                    else:
                        feedback += "Raise both hands above your head."
                # Add more poses here as needed

                if pose_correct and not self.in_pose:
                    self.narrate("Pose correct. Hold it.")
                    self.in_pose = True
                elif not pose_correct and self.in_pose:
                    self.narrate("Pose lost. Try again.")
                    self.in_pose = False
            else:
                # Default feedback if not in yoga flow
                if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                    if abs(left_shoulder.y - right_shoulder.y) < 0.05 and abs(left_hip.y - right_hip.y) < 0.05:
                        feedback = "Perfect: Both hands up and standing straight!"
                    else:
                        feedback = "Hands up! Try to stand straight."
                elif left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
                    feedback = "Raise both hands above shoulders."
                else:
                    feedback = "Try raising your hands!"
                self.narrate(feedback)

        self.landmark_data.extend(landmarks_this_frame)

        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

        self.plot3d.update(coords_for_3d)

        c_time = time.time()
        fps = 1 / (c_time - self.fps_time + 1e-6)
        self.fps_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if self.recording and self.video_writer:
            resized = cv2.resize(frame, (640, 480))
            self.video_writer.write(resized)

        small_frame = cv2.resize(frame, (320, 240))
        max_gif_frames = 100
        if len(self.frames) < max_gif_frames:
            self.frames.append(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

        img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))
        self.current_frame = frame.copy()

        self.feedback_label.setText(feedback)

    def closeEvent(self, event):
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if self.music_playing:
            pygame.mixer.music.stop()
        self.plot3d.running = False
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = NeonPoseApp()
    window.show()
    sys.exit(app.exec_())