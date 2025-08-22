# vehicle_entry_app.py
import sys
import os
import csv
import cv2
import threading
from datetime import datetime
from collections import defaultdict

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QTextEdit,
    QMessageBox,
    QGridLayout,
    QInputDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

from ultralytics import YOLO


class ClickableLabel(QLabel):
    """QLabel that emits click coordinates (in label widget coords)."""
    clicked = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class VehicleEntryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Entry Detection App")
        self.setGeometry(120, 80, 1200, 880)

        # --- State ---
        self.video_paths = []
        self.log_folder = ""
        self.main_log_folder = ""
        self.entry_line = None               # [(x1,y1),(x2,y2)] in frame coords
        self.frame_for_line = None
        self.click_points = []
        self.line_drawing_enabled = False

        self.stop_flag = False
        self.pause_flag = threading.Event()
        self.pause_flag.set()

        # Counters (per-video)
        self.total_cars = 0
        self.total_motorcycles = 0
        self.adjusted_cars = 0
        self.adjusted_motorcycles = 0

        # Track bookkeeping:
        # track_memory[id] = {'prev_side': float or None, 'counted': bool}
        self.track_memory = {}

        # model path (keep so we can reload between videos to clear tracker state)
        self.model_path = "yolov8m.pt"
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.model = None
            print(f"[WARN] could not load model at init: {e}")

        # detection params
        self.conf = 0.15  # you can lower if motorcycles are missed
        self.iou = 0.45

        # debugging / speed
        self.frame_skip = 5  # process every Nth frame (set 1 for every frame)
        # class map (populated after model load)
        self.class_map = {}
        if self.model:
            self.class_map = self.model.names

        # --- UI widgets ---
        self.upload_btn = QPushButton("Upload Video(s)")
        self.folder_btn = QPushButton("Select Log Folder")
        self.start_btn = QPushButton("Start Detection")
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")

        for btn in (self.upload_btn, self.folder_btn, self.start_btn,
                    self.pause_btn, self.resume_btn, self.stop_btn):
            btn.setFixedWidth(150)

        # Preview area (Clickable)
        self.preview_label = ClickableLabel()
        self.preview_label.setFixedSize(960, 540)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.clicked.connect(self._preview_click_for_line_coords)

        # Logs
        self.log_screen = QTextEdit()
        self.log_screen.setReadOnly(True)
        self.log_screen.setFixedHeight(160)

        # Counter labels
        self.total_cars_lbl = QLabel("Total Cars\n0")
        self.total_cars_lbl.setAlignment(Qt.AlignCenter)
        self.total_cars_lbl.setStyleSheet("font-size:18px; font-weight:700; color: white;")

        self.total_moto_lbl = QLabel("Total Motorcycles\n0")
        self.total_moto_lbl.setAlignment(Qt.AlignCenter)
        self.total_moto_lbl.setStyleSheet("font-size:18px; font-weight:700; color: white;")

        self.adjusted_cars_lbl = QLabel("Adjusted Cars\n0")
        self.adjusted_cars_lbl.setAlignment(Qt.AlignCenter)
        self.adjusted_cars_lbl.setStyleSheet("font-size:18px; font-weight:700; color: white;")

        self.adjusted_moto_lbl = QLabel("Adjusted Motorcycles\n0")
        self.adjusted_moto_lbl.setAlignment(Qt.AlignCenter)
        self.adjusted_moto_lbl.setStyleSheet("font-size:18px; font-weight:700; color: white;")

        # Inline +/- buttons next to adjusted counters
        self.adj_car_minus = QPushButton("−")
        self.adj_car_plus = QPushButton("+")
        self.adj_moto_minus = QPushButton("−")
        self.adj_moto_plus = QPushButton("+")

        for b in (self.adj_car_minus, self.adj_car_plus, self.adj_moto_minus, self.adj_moto_plus):
            b.setFixedSize(36, 28)

        # --- Layouts ---
        top_btn_layout = QHBoxLayout()
        top_btn_layout.setSpacing(20)
        top_btn_layout.addStretch()
        top_btn_layout.addWidget(self.upload_btn)
        top_btn_layout.addWidget(self.folder_btn)
        top_btn_layout.addWidget(self.start_btn)
        top_btn_layout.addWidget(self.pause_btn)
        top_btn_layout.addWidget(self.resume_btn)
        top_btn_layout.addWidget(self.stop_btn)
        top_btn_layout.addStretch()

        counters_grid = QGridLayout()
        counters_grid.setHorizontalSpacing(40)
        counters_grid.setVerticalSpacing(8)
        counters_grid.addWidget(self.total_cars_lbl, 0, 0)
        counters_grid.addWidget(self.total_moto_lbl, 0, 1)

        adj_car_row = QHBoxLayout()
        adj_car_row.setAlignment(Qt.AlignCenter)
        adj_car_row.addWidget(self.adj_car_minus)
        adj_car_row.addSpacing(8)
        adj_car_row.addWidget(self.adjusted_cars_lbl)
        adj_car_row.addSpacing(8)
        adj_car_row.addWidget(self.adj_car_plus)

        adj_moto_row = QHBoxLayout()
        adj_moto_row.setAlignment(Qt.AlignCenter)
        adj_moto_row.addWidget(self.adj_moto_minus)
        adj_moto_row.addSpacing(8)
        adj_moto_row.addWidget(self.adjusted_moto_lbl)
        adj_moto_row.addSpacing(8)
        adj_moto_row.addWidget(self.adj_moto_plus)

        counters_grid.addLayout(adj_car_row, 1, 0)
        counters_grid.addLayout(adj_moto_row, 1, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_btn_layout)
        main_layout.addSpacing(12)
        counters_container = QHBoxLayout()
        counters_container.addStretch()
        counters_container.addLayout(counters_grid)
        counters_container.addStretch()
        main_layout.addLayout(counters_container)
        main_layout.addSpacing(12)

        preview_container = QHBoxLayout()
        preview_container.addStretch()
        preview_container.addWidget(self.preview_label)
        preview_container.addStretch()
        main_layout.addLayout(preview_container)

        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Logs:"))
        main_layout.addWidget(self.log_screen)

        self.setLayout(main_layout)

        # --- Connections ---
        self.upload_btn.clicked.connect(self.upload_videos)
        self.folder_btn.clicked.connect(self.select_folder)
        self.start_btn.clicked.connect(self.start_detection)
        self.pause_btn.clicked.connect(self.pause_detection)
        self.resume_btn.clicked.connect(self.resume_detection)
        self.stop_btn.clicked.connect(self.stop_detection)

        self.adj_car_plus.clicked.connect(lambda: self.user_adjust('car', 1))
        self.adj_car_minus.clicked.connect(lambda: self.user_adjust('car', -1))
        self.adj_moto_plus.clicked.connect(lambda: self.user_adjust('moto', 1))
        self.adj_moto_minus.clicked.connect(lambda: self.user_adjust('moto', -1))

        # initial UI state
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

    # ------------------- UI callbacks -------------------
    def upload_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Video(s)", "", "Videos (*.mp4 *.avi *.mov)")
        # Also allow selecting a folder of videos
        folder = None
        if not paths:
            # if no files selected, prompt folder optionally
            folder = QFileDialog.getExistingDirectory(self, "Or Select Folder of Videos (Cancel to skip)")
        else:
            # give user option to add folder as well
            add_folder = QMessageBox.question(self, "Add Folder?", "Do you want to add a folder of videos as well?",
                                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if add_folder == QMessageBox.Yes:
                folder = QFileDialog.getExistingDirectory(self, "Select Folder of Videos")

        if folder:
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    full = os.path.join(folder, f)
                    if full not in paths:
                        paths.append(full)

        if paths:
            # remove duplicates while preserving order
            seen = set()
            ordered = []
            for p in paths:
                if p not in seen:
                    ordered.append(p)
                    seen.add(p)
            self.video_paths = ordered
            self.log_screen.append(f"[INFO] {len(self.video_paths)} video(s) selected:")
            for i, p in enumerate(self.video_paths, 1):
                self.log_screen.append(f"  {i}. {p}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Log Folder")
        if folder:
            date_str, ok1 = QInputDialog.getText(self, "Date", "Enter date (DD-MM-YYYY):")
            if not ok1 or not date_str.strip():
                self.log_screen.append("[ERROR] Date input canceled or empty.")
                return
            site_name, ok2 = QInputDialog.getText(self, "Site Name", "Enter site name:")
            if not ok2 or not site_name.strip():
                self.log_screen.append("[ERROR] Site name input canceled or empty.")
                return

            # Store for later use
            self.date_str = date_str.strip()
            self.site_name = site_name.strip()

            safe_site = site_name.strip().replace(" ", "_")
            folder_name = f"{date_str.strip()}_{safe_site}"
            self.main_log_folder = os.path.join(folder, folder_name)
            os.makedirs(self.main_log_folder, exist_ok=True)
            self.log_folder = folder
            self.log_screen.append(f"[INFO] Main log folder: {self.main_log_folder}")


    def start_detection(self):
        if not self.video_paths or not self.main_log_folder:
            QMessageBox.warning(self, "Missing Input", "Please select video(s) and a log folder (and provide date & site).")
            return

        # Load first frame of the first video so user can draw entry line
        cap = cv2.VideoCapture(self.video_paths[0])
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            self.log_screen.append("[ERROR] Failed to read frame for line drawing.")
            return

        self.frame_for_line = frame.copy()
        self.preview_label.setPixmap(self.cv_to_pixmap(self.frame_for_line))
        self.click_points = []
        self.line_drawing_enabled = True
        self.preview_label.setCursor(Qt.CrossCursor)
        self.log_screen.append("[INFO] Click two points inside the preview to draw entry line...")

    def _preview_click_for_line_coords(self, lx, ly):
        """Handle clicks from ClickableLabel (label coords), map to frame coords, and collect two points."""
        if not self.line_drawing_enabled:
            return
        pix = self.preview_label.pixmap()
        if pix is None or self.frame_for_line is None:
            return

        label_w, label_h = self.preview_label.width(), self.preview_label.height()
        pix_w, pix_h = pix.width(), pix.height()
        offset_x = max((label_w - pix_w) // 2, 0)
        offset_y = max((label_h - pix_h) // 2, 0)

        # x,y relative to the pixmap inside the label
        px = min(max(lx - offset_x, 0), pix_w - 1)
        py = min(max(ly - offset_y, 0), pix_h - 1)

        frame_h, frame_w = self.frame_for_line.shape[:2]
        scale_x = frame_w / pix_w
        scale_y = frame_h / pix_h

        real_x = int(px * scale_x)
        real_y = int(py * scale_y)

        self.click_points.append((real_x, real_y))
        # show small marker while selecting (draw temp)
        tmp = self.frame_for_line.copy()
        for pt in self.click_points:
            cv2.circle(tmp, pt, 6, (0, 255, 255), -1)
        self.preview_label.setPixmap(self.cv_to_pixmap(tmp))

        if len(self.click_points) == 2:
            p1, p2 = self.click_points
            cv2.line(self.frame_for_line, p1, p2, (0, 255, 0), 2)
            self.preview_label.setPixmap(self.cv_to_pixmap(self.frame_for_line))
            self.entry_line = [p1, p2]
            self.log_screen.append(f"[INFO] Entry line set: {p1} -> {p2}")
            # disable drawing
            self.line_drawing_enabled = False
            self.preview_label.setCursor(Qt.ArrowCursor)

            # reset counters/bookkeeping and start detection
            self.total_cars = 0
            self.total_motorcycles = 0
            self.adjusted_cars = 0
            self.adjusted_motorcycles = 0
            self.track_memory.clear()
            self.update_counter_labels()
            self._start_detection_thread()

    def pause_detection(self):
        self.pause_flag.clear()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.log_screen.append("⏯️ Detection paused.")

    def resume_detection(self):
        self.pause_flag.set()
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.log_screen.append("⏯️ Detection resumed.")

    def stop_detection(self):
        self.stop_flag = True
        self.pause_flag.set()
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.log_screen.append("🛑 Detection stopped.")

    def user_adjust(self, kind, delta):
        """Manual +/- for adjusted counts (inline buttons)."""
        if kind == 'car':
            self.adjusted_cars = max(0, self.adjusted_cars + delta)
        else:
            self.adjusted_motorcycles = max(0, self.adjusted_motorcycles + delta)
        self.update_counter_labels()

    # ------------------- Helpers -------------------
    def cv_to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
        return pix

    def update_counter_labels(self):
        self.total_cars_lbl.setText(f"Total Cars\n{self.total_cars}")
        self.total_moto_lbl.setText(f"Total Motorcycles\n{self.total_motorcycles}")
        self.adjusted_cars_lbl.setText(f"Adjusted Cars\n{self.adjusted_cars}")
        self.adjusted_moto_lbl.setText(f"Adjusted Motorcycles\n{self.adjusted_motorcycles}")

    @staticmethod
    def side_of_line(pt, A, B):
        # cross product (B-A) x (pt-A)
        return (B[0] - A[0]) * (pt[1] - A[1]) - (B[1] - A[1]) * (pt[0] - A[0])

    def _start_detection_thread(self):
        self.stop_flag = False
        self.pause_flag.set()
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        threading.Thread(target=self.run_detection, daemon=True).start()

    def run_detection(self):
        """
        Main detection loop:
        - uses model.track(..., tracker="bytetrack.yaml") for consistent IDs
        - increments totals & adjusted counters when a track crosses the entry line for the first time
        - writes per-video CSV rows for crossings
        - writes per-video summary and a global total_summary.csv
        """
        # Ensure model loaded (lazy load)
        if self.model is None:
            try:
                self.model = YOLO(self.model_path)
                self.class_map = self.model.names
            except Exception as e:
                self.log_screen.append(f"[ERROR] Could not load YOLO model: {e}")
                return

        # compute class id sets
        try:
            cm = self.model.names
            self.class_map = cm
        except Exception:
            cm = self.class_map or {}

        car_ids = {cid for cid, name in cm.items() if 'car' in name.lower()}
        moto_ids = {cid for cid, name in cm.items() if any(term in name.lower() for term in ['motorcycle', 'motorbike', 'moped', 'scooter', 'bike']) and 'bicycle' not in name.lower()}
        track_classes = sorted(list(car_ids | moto_ids)) or None
        if not moto_ids:
            self.log_screen.append("[WARN] Motorcycle class ids not found; using name heuristics and area fallback.")

        # global totals
        global_total_cars = 0
        global_total_motorcycles = 0
        global_adjusted_cars = 0
        global_adjusted_motorcycles = 0

        # iterate videos
        for vid_idx, video_path in enumerate(self.video_paths):
            if self.stop_flag:
                break

            # Record start time for this video
            start_time = datetime.now().strftime("%H:%M:%S")

            # reset per-video counters
            self.total_cars = 0
            self.total_motorcycles = 0
            self.adjusted_cars = 0
            self.adjusted_motorcycles = 0
            self.track_memory.clear()
            self.update_counter_labels()

            self.log_screen.append(f"[INFO] Starting video {vid_idx+1}/{len(self.video_paths)}: {os.path.basename(video_path)}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log_screen.append(f"[ERROR] Cannot open {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            image_dir = os.path.join(self.main_log_folder, f"{video_name}_images")
            os.makedirs(image_dir, exist_ok=True)
            log_path = os.path.join(self.main_log_folder, f"{video_name}_log.csv")

            # detection histogram (debug)
            detection_hist = defaultdict(int)

            with open(log_path, "w", newline="") as log_file:
                csv_writer = csv.writer(log_file)
                csv_writer.writerow(["TrackID", "Timestamp", "Class", "ImagePath"])

                frame_count = 0
                while cap.isOpened():
                    if self.stop_flag:
                        break
                    self.pause_flag.wait()

                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                    # frame skipping for speed
                    if frame_count % self.frame_skip != 0:
                        # still draw entry line for preview
                        disp = frame.copy()
                        if self.entry_line:
                            cv2.line(disp, self.entry_line[0], self.entry_line[1], (0, 255, 0), 2)
                        self.preview_label.setPixmap(self.cv_to_pixmap(disp))
                        continue

                    # model.track with classes filter if available
                    try:
                        if track_classes:
                            results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                                       classes=track_classes, conf=self.conf, iou=self.iou)
                        else:
                            results = self.model.track(source=frame, persist=True, tracker="bytetrack.yaml",
                                                       conf=self.conf, iou=self.iou)
                    except Exception as e:
                        self.log_screen.append(f"[ERROR] model.track() failed: {e}")
                        results = None

                    if not results or len(results) == 0:
                        disp = frame.copy()
                        if self.entry_line:
                            cv2.line(disp, self.entry_line[0], self.entry_line[1], (0, 255, 0), 2)
                        self.preview_label.setPixmap(self.cv_to_pixmap(disp))
                        continue

                    boxes = results[0].boxes
                    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
                        disp = frame.copy()
                        if self.entry_line:
                            cv2.line(disp, self.entry_line[0], self.entry_line[1], (0, 255, 0), 2)
                        self.preview_label.setPixmap(self.cv_to_pixmap(disp))
                        continue

                    xyxy = boxes.xyxy.cpu().numpy()       # Nx4
                    clsids = boxes.cls.cpu().numpy().astype(int)  # N
                    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                    for i_det in range(xyxy.shape[0]):
                        x1, y1, x2, y2 = map(int, xyxy[i_det])
                        clsid = int(clsids[i_det])
                        obj_id = int(ids[i_det]) if ids is not None else None
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        class_name = self.class_map.get(clsid, f"class_{clsid}")
                        detection_hist[class_name] += 1

                        # heuristics for vehicle type
                        is_car = (clsid in car_ids) or ('car' in class_name.lower())
                        is_moto = (clsid in moto_ids) or (any(term in class_name.lower() for term in ['motorcycle', 'motorbike', 'moped', 'scooter', 'bike']) and 'bicycle' not in class_name.lower())

                        # special: bicycle but large -> treat as motorcycle
                        if class_name.lower() == 'bicycle':
                            box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                            if box_area > 2500:  # tweak threshold for your camera
                                is_moto = True

                        if not (is_car or is_moto):
                            # skip other classes
                            continue

                        # compute side-of-line
                        if self.entry_line and len(self.entry_line) == 2:
                            side_now = self.side_of_line((cx, cy), self.entry_line[0], self.entry_line[1])
                        else:
                            side_now = None

                        # init track memory
                        if obj_id is not None and obj_id not in self.track_memory:
                            self.track_memory[obj_id] = {'prev_side': side_now, 'counted': False}

                        prev_side = None
                        counted = False
                        if obj_id is not None:
                            prev_side = self.track_memory[obj_id].get('prev_side', None)
                            counted = self.track_memory[obj_id].get('counted', False)

                        # crossing detection: sign change and not yet counted
                        if (
                            prev_side is not None
                            and side_now is not None
                            and prev_side < 0        # started on the right
                            and side_now > 0         # moved to the left
                            and not counted
                        ):
                            # timestamp based on video time (frame_count / fps)
                            video_time_seconds = frame_count / fps if fps > 0 else 0.0
                            hours = int(video_time_seconds // 3600)
                            minutes = int((video_time_seconds % 3600) // 60)
                            seconds = video_time_seconds % 60
                            timestamp = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

                            safe_ts = timestamp.replace(":", "-")
                            # safe crop bounds
                            y1c, y2c = max(0, y1), min(frame.shape[0], y2)
                            x1c, x2c = max(0, x1), min(frame.shape[1], x2)
                            if y2c > y1c and x2c > x1c:
                                try:
                                    cv2.imwrite(os.path.join(image_dir, f"{obj_id}_{safe_ts}.jpg"), frame[y1c:y2c, x1c:x2c])
                                except Exception:
                                    cv2.imwrite(os.path.join(image_dir, f"{obj_id}_{safe_ts}.jpg"), frame)
                            else:
                                cv2.imwrite(os.path.join(image_dir, f"{obj_id}_{safe_ts}.jpg"), frame)

                            image_path = os.path.join(image_dir, f"{obj_id}_{safe_ts}.jpg")

                            if is_car:
                                self.total_cars += 1
                                self.adjusted_cars += 1
                                global_total_cars += 1
                                global_adjusted_cars += 1
                                csv_writer.writerow([obj_id, timestamp, "car", image_path])
                                self.log_screen.append(f"[CROSS] car id={obj_id} at {timestamp}")
                            elif is_moto:
                                self.total_motorcycles += 1
                                self.adjusted_motorcycles += 1
                                global_total_motorcycles += 1
                                global_adjusted_motorcycles += 1
                                csv_writer.writerow([obj_id, timestamp, "motorcycle", image_path])
                                self.log_screen.append(f"[CROSS] motorcycle id={obj_id} at {timestamp}")

                            # mark counted
                            if obj_id is not None:
                                self.track_memory[obj_id]['counted'] = True

                        # update prev_side for next frame
                        if obj_id is not None:
                            self.track_memory[obj_id]['prev_side'] = side_now

                        # draw bounding box & label for preview
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        id_text = f" ID:{obj_id}" if obj_id is not None else ""
                        cv2.putText(frame, f"{class_name}{id_text}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 0), 1)

                    # draw entry line
                    if self.entry_line and len(self.entry_line) == 2:
                        cv2.line(frame, self.entry_line[0], self.entry_line[1], (0, 255, 0), 2)

                    # draw counters
                    cv2.putText(frame, f"Total Cars: {self.total_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 255), 2)
                    cv2.putText(frame, f"Total Motorcycles: {self.total_motorcycles}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # update GUI preview + labels
                    self.preview_label.setPixmap(self.cv_to_pixmap(frame))
                    self.update_counter_labels()

                # end frames for this video

            # log detection histogram for this video (debug)
            if detection_hist:
                self.log_screen.append("[DEBUG] Detection histogram for this video:")
                for cname, cnt in detection_hist.items():
                    self.log_screen.append(f"  {cname}: {cnt}")

            cap.release()

            # Append per-video summary to CSV (already open above but now append)
            try:
                with open(log_path, "a", newline="") as log_file:
                    csv_writer = csv.writer(log_file)
                    csv_writer.writerow([])
                    csv_writer.writerow(["--- SUMMARY ---"])
                    csv_writer.writerow(["Total Cars", self.total_cars])
                    csv_writer.writerow(["Adjusted Cars", self.adjusted_cars])
                    csv_writer.writerow(["Total Motorcycles", self.total_motorcycles])
                    csv_writer.writerow(["Adjusted Motorcycles", self.adjusted_motorcycles])
                self.log_screen.append(f"[SUMMARY] Video {vid_idx+1}: Cars={self.total_cars}, AdjCars={self.adjusted_cars}, Motos={self.total_motorcycles}, AdjMotos={self.adjusted_motorcycles}")
            except Exception as e:
                self.log_screen.append(f"[ERROR] Could not write summary to CSV: {e}")

            # Write to total_summary.csv after each video
            end_time = datetime.now().strftime("%H:%M:%S")
            total_count = self.total_cars + self.total_motorcycles
            summary_path = os.path.join(self.main_log_folder, "total_summary.csv")
            file_exists = os.path.isfile(summary_path)
            import csv as _csv  # in case csv is shadowed
            with open(summary_path, "a", newline="") as f:
                writer = _csv.writer(f)
                if not file_exists:
                    writer.writerow(["SITE NAME", "DATE", "VIDEO-ID", "START-TIME", "END-TIME", "CAR", "MOTORCYCLE", "TOTAL"])
                video_id = os.path.splitext(os.path.basename(video_path))[0]
                writer.writerow([self.site_name, self.date_str, video_id, start_time, end_time, self.total_cars, self.total_motorcycles, total_count])
            self.log_screen.append(f"[SUMMARY] {video_id} - Saved to total_summary.csv")

            # reload model to clear tracker state before next video (optional)
            try:
                self.model = YOLO(self.model_path)
                self.class_map = self.model.names
            except Exception as e:
                self.log_screen.append(f"[WARN] Could not reload model between videos: {e}")

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.log_screen.append("[DONE] Processing complete.")

# -------------------- run app --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VehicleEntryApp()
    window.show()
    sys.exit(app.exec_())