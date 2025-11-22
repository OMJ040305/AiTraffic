import cv2
import time
import threading
import numpy as np

import config as cfg
from detector import VehicleDetector
import visualizer as vis


class TrafficLightSystem:
    def __init__(self):
        self.cameras = {}

        # Estados del sistema
        self.traffic_states = {ch: 'red' for ch in cfg.CAMERA_CHANNELS}
        self.arrow_states = {ch: 'red' for ch in cfg.CAMERA_CHANNELS}
        self.detection_counts = {ch: {'main': 0, 'arrow': 0} for ch in cfg.CAMERA_CHANNELS}
        self.camera_status = {ch: 'unknown' for ch in cfg.CAMERA_CHANNELS}
        self.camera_failures = {ch: 0 for ch in cfg.CAMERA_CHANNELS}
        self.system_mode = {ch: 'INTELLIGENT' for ch in cfg.CAMERA_CHANNELS}
        self.last_frame_time = {ch: time.time() for ch in cfg.CAMERA_CHANNELS}

        # Caché visual y optimización
        self.last_detections = {ch: [] for ch in cfg.CAMERA_CHANNELS}
        self.frame_counter = 0
        self.detection_interval = 4

        # --- GESTIÓN DE ZONAS EN VIVO ---
        self.live_zones = {}
        for i, ch in enumerate(cfg.CAMERA_CHANNELS):
            mz, az = cfg.get_zones(i)
            self.live_zones[ch] = {
                'main': mz[0] if len(mz) > 0 else [],
                'arrow': az[0] if len(az) > 0 else []
            }

        # --- VARIABLES DE EDICIÓN ---
        self.is_editing = False
        self.edit_channel = None
        self.edit_zone_type = 'main'
        self.edit_points = []
        self.click_cooldown = 0

        # Detector
        self.detector = VehicleDetector()

        # Control de secuencia
        self.current_phase = 0
        self.phase_start_time = time.time()
        self.sequence_lock = threading.Lock()
        self.running = True

        # Hilos
        self.threads = []
        self.threads.append(threading.Thread(target=self.intelligent_control, daemon=True))
        self.threads.append(threading.Thread(target=self.standard_control, daemon=True))
        self.threads.append(threading.Thread(target=self.monitor_cameras, daemon=True))

        for t in self.threads: t.start()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = time.time()
            if current_time - self.click_cooldown < 0.3: return
            self.click_cooldown = current_time

            # LOGICA 1: MODO GRID (Seleccionar cámara)
            if not self.is_editing:
                # Si hacen click en el menú lateral (x > 960), ignorar
                if x > 960: return

                col = 0 if x < 480 else 1
                row = 0 if y < 360 else 1
                idx = row * 2 + col

                if idx < len(cfg.CAMERA_CHANNELS):
                    target_ch = cfg.CAMERA_CHANNELS[idx]
                    print(f"✏️ Seleccionada Cámara {idx} ({target_ch}) para edición")
                    self.edit_channel = target_ch
                    self.edit_points = []
                    self.is_editing = True
                    self.edit_zone_type = 'main'

                    # LOGICA 2: MODO EDICIÓN
            else:
                # Si hacen click en el menú lateral (x > 640 aprox para video individual), ignorar
                if x < 640:
                    self.edit_points.append([x, y])
                    print(f"📍 Punto: {x},{y}")

    # --- MÉTODOS DE CONTROL (SIN CAMBIOS) ---
    def has_vehicles(self, channel, type='any'):
        if self.system_mode[channel] != 'INTELLIGENT': return True
        cnt = self.detection_counts[channel]
        if type == 'arrow': return cnt['arrow'] > 0
        return cnt['main'] > 0 or cnt['arrow'] > 0

    def should_skip_phase(self, phase):
        if phase == 0:
            e, o = cfg.CAMERA_CHANNELS[cfg.ESTE_IDX], cfg.CAMERA_CHANNELS[cfg.OESTE_IDX]
            return not (self.has_vehicles(e, 'arrow') or self.has_vehicles(o, 'arrow'))
        elif phase == 2:
            return not self.has_vehicles(cfg.CAMERA_CHANNELS[cfg.NORTE_IDX])
        elif phase == 3:
            return not self.has_vehicles(cfg.CAMERA_CHANNELS[cfg.SUR_IDX])
        return False

    def set_lights(self, phase, color='green'):
        for ch in cfg.CAMERA_CHANNELS:
            self.traffic_states[ch] = 'red'
            self.arrow_states[ch] = 'red'

        if color != 'red':
            if phase == 0:
                self.arrow_states[cfg.CAMERA_CHANNELS[cfg.ESTE_IDX]] = color
                self.arrow_states[cfg.CAMERA_CHANNELS[cfg.OESTE_IDX]] = color
            elif phase == 1:
                self.traffic_states[cfg.CAMERA_CHANNELS[cfg.ESTE_IDX]] = color
                self.traffic_states[cfg.CAMERA_CHANNELS[cfg.OESTE_IDX]] = color
            elif phase == 2:
                self.traffic_states[cfg.CAMERA_CHANNELS[cfg.NORTE_IDX]] = color
            elif phase == 3:
                self.traffic_states[cfg.CAMERA_CHANNELS[cfg.SUR_IDX]] = color

    def intelligent_control(self):
        while self.running:
            with self.sequence_lock:
                if not any(self.system_mode[ch] == 'INTELLIGENT' for ch in cfg.CAMERA_CHANNELS):
                    time.sleep(1)
                    continue

                elapsed = time.time() - self.phase_start_time
                phase_time = cfg.PHASE_TIMES[self.current_phase]

                if elapsed >= phase_time - cfg.YELLOW_TIME:
                    if elapsed < phase_time:
                        self.set_lights(self.current_phase, 'yellow')
                    else:
                        next_ph = (self.current_phase + 1) % 4
                        skipped = 0
                        while self.should_skip_phase(next_ph) and skipped < 4:
                            print(f"⏭️ Saltando fase {next_ph}")
                            next_ph = (next_ph + 1) % 4
                            skipped += 1

                        if skipped == 4:
                            if self.current_phase != 1:
                                self.current_phase = 1
                                self.set_lights(1, 'green')
                        else:
                            self.current_phase = next_ph
                            self.set_lights(next_ph, 'green')

                        self.phase_start_time = time.time()
            time.sleep(0.5)

    def standard_control(self):
        durations = [cfg.PHASE_TIMES[i] for i in range(4)]
        start_t = time.time()
        curr_ph = 0

        while self.running:
            std_cams = [ch for ch in cfg.CAMERA_CHANNELS if self.system_mode[ch] in ['STANDARD', 'FALLBACK']]
            if not std_cams:
                time.sleep(1)
                continue

            elapsed = time.time() - start_t
            total_ph_time = durations[curr_ph]
            is_yellow = elapsed >= (total_ph_time - cfg.YELLOW_TIME)

            if elapsed >= total_ph_time:
                curr_ph = (curr_ph + 1) % 4
                start_t = time.time()
                is_yellow = False

            for ch in std_cams:
                idx = cfg.CAMERA_CHANNELS.index(ch)
                t_color, a_color = 'red', 'red'

                if curr_ph == 0 and idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
                    a_color = 'yellow' if is_yellow else 'green'
                elif curr_ph == 1 and idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
                    t_color = 'yellow' if is_yellow else 'green'
                elif curr_ph == 2 and idx == cfg.NORTE_IDX:
                    t_color = 'yellow' if is_yellow else 'green'
                elif curr_ph == 3 and idx == cfg.SUR_IDX:
                    t_color = 'yellow' if is_yellow else 'green'

                self.traffic_states[ch] = t_color
                self.arrow_states[ch] = a_color
            time.sleep(0.2)

    def process_camera(self, channel, frame):
        self.last_frame_time[channel] = time.time()
        if self.system_mode[channel] != 'INTELLIGENT': return frame

        if self.frame_counter % self.detection_interval == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))

            bboxes = self.detector.detect(small)
            if len(bboxes) > 0:
                bboxes = (bboxes / 0.5).astype(int)
            else:
                bboxes = np.empty((0, 4))

            self.last_detections[channel] = bboxes

            current_rect = [self.live_zones[channel]['main']]
            current_arrow = [self.live_zones[channel]['arrow']]

            c_main, c_arrow = 0, 0
            for box in bboxes:
                xc, yc = self.detector.get_center(box)
                if self.detector.is_valid_detection(xc, yc, current_rect): c_main += 1
                if self.detector.is_valid_detection(xc, yc, current_arrow): c_arrow += 1

            self.detection_counts[channel] = {'main': c_main, 'arrow': c_arrow}

        for box in self.last_detections[channel]:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            xc, yc = self.detector.get_center(box)
            cv2.circle(frame, (xc, yc), 3, (0, 0, 255), -1)

        return frame

    def monitor_cameras(self):
        while self.running:
            now = time.time()
            for ch in cfg.CAMERA_CHANNELS:
                if self.camera_status[ch] == 'active' and (now - self.last_frame_time[ch] > cfg.CAMERA_TIMEOUT):
                    self.camera_status[ch] = 'failed'
                    self.system_mode[ch] = 'STANDARD'
                    self.attempt_reconnect(ch)
            time.sleep(2)

    def attempt_reconnect(self, channel):
        try:
            if channel in self.cameras: self.cameras[channel].release()
            cap = cv2.VideoCapture(channel)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.cameras[channel] = cap
                    self.camera_status[channel] = 'active'
                    self.system_mode[channel] = 'INTELLIGENT'
                    self.last_frame_time[channel] = time.time()
                    print(f"♻️ Cámara {channel} conectada.")
        except:
            pass

    def initialize_cameras(self):
        print("Conectando cámaras...")
        for ch in cfg.CAMERA_CHANNELS:
            self.attempt_reconnect(ch)

    def run(self):
        print("=== SISTEMA DE TRAFICO AI INICIADO ===")
        self.initialize_cameras()

        window_name = 'Sistema Semaforo'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            self.frame_counter += 1

            if self.is_editing and self.edit_channel in self.cameras:
                # --- MODO EDICIÓN (VIDEO + MENÚ DERECHO) ---
                ret, raw = self.cameras[self.edit_channel].read()
                if ret:
                    edit_frame = vis.draw_edit_mode(raw.copy(), self.edit_points,
                                                    f"EDITANDO: {self.edit_channel}",
                                                    self.edit_zone_type)
                    cv2.imshow(window_name, edit_frame)

                k = cv2.waitKey(1) & 0xFF
                if k == 27:  # ESC
                    self.is_editing = False
                elif k == ord('z'):  # Undo
                    if self.edit_points: self.edit_points.pop()
                elif k == ord('t'):  # Toggle
                    self.edit_zone_type = 'arrow' if self.edit_zone_type == 'main' else 'main'
                elif k == ord('s'):  # Save
                    new_zone = np.array(self.edit_points) if len(self.edit_points) > 2 else []
                    self.live_zones[self.edit_channel][self.edit_zone_type] = new_zone
                    self.is_editing = False

            else:
                # --- MODO DASHBOARD (GRID + MENÚ PRINCIPAL) ---
                frames_list = []
                for i, ch in enumerate(cfg.CAMERA_CHANNELS):
                    frame = np.zeros((360, 480, 3), dtype=np.uint8)
                    if ch in self.cameras and self.cameras[ch].isOpened():
                        ret, raw = self.cameras[ch].read()
                        if ret:
                            frame = self.process_camera(ch, raw)
                            state = {
                                'mode': self.system_mode[ch],
                                'status': self.camera_status[ch],
                                'traffic_color': self.traffic_states[ch],
                                'arrow_color': self.arrow_states[ch],
                                'counts': self.detection_counts[ch],
                                'zones': (self.live_zones[ch]['main'], self.live_zones[ch]['arrow'])
                            }
                            frame = vis.add_overlay(frame, ch, cfg.CAMERA_NAMES[i], i, state)
                    frames_list.append(cv2.resize(frame, (480, 360)))

                top = np.hstack([frames_list[0], frames_list[1]])
                bot = np.hstack([frames_list[2], frames_list[3]])
                grid = np.vstack([top, bot])

                # PREPARAR DATOS PARA EL DASHBOARD
                dashboard_info = {
                    'phase_idx': self.current_phase,
                    'active_cams': sum(1 for s in self.camera_status.values() if s == 'active'),
                    'intelligent_cams': sum(1 for m in self.system_mode.values() if m == 'INTELLIGENT')
                }

                # COMBINAR GRID + DASHBOARD
                final_view = vis.draw_dashboard(grid, dashboard_info)

                cv2.imshow(window_name, final_view)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.running = False
        cv2.destroyAllWindows()
        for cap in self.cameras.values(): cap.release()


if __name__ == '__main__':
    TrafficLightSystem().run()