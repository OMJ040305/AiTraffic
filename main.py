import cv2
import time
import threading
import numpy as np
import math
import datetime
import os

# Importar módulos propios
import config as cfg
from detector import VehicleDetector
from tracker import EuclideanDistTracker
import visualizer as vis
from stats import StatsManager  # <--- NUEVO IMPORT PARA ESTADÍSTICAS


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

        # --- RASTREO Y DETECCIÓN DE INCIDENTES ---
        self.trackers = {ch: EuclideanDistTracker() for ch in cfg.CAMERA_CHANNELS}

        # Estructura de datos:
        # { id: { 'last_pos': (x,y), 'accumulated_time': float, 'last_update_time': float, 'incident_type': str, 'alert_sent': bool } }
        self.vehicle_data = {ch: {} for ch in cfg.CAMERA_CHANNELS}

        # --- GESTOR DE ESTADÍSTICAS ---
        self.stats_manager = StatsManager()  # <--- INICIALIZACIÓN

        # --- CONFIGURACIÓN DE INCIDENTES ---
        self.STOP_THRESHOLD = 15
        self.ACCIDENT_TIME = 120.0  # 2 MINUTOS
        self.COLLISION_DIST = 150  # Distancia para colisión

        # Caché visual
        self.last_detections = {ch: [] for ch in cfg.CAMERA_CHANNELS}
        self.frame_counter = 0
        self.detection_interval = 3

        # --- GESTIÓN DE ZONAS EN VIVO ---
        self.live_zones = {}
        for i, ch in enumerate(cfg.CAMERA_CHANNELS):
            mz, az = cfg.get_zones(i)
            self.live_zones[ch] = {
                'main': mz[0] if len(mz) > 0 else [],
                'arrow': az[0] if len(az) > 0 else []
            }

        # Variables de Edición
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

    # --- GESTIÓN DE EVIDENCIAS E INFORMES ---
    def handle_incident_log(self, channel, vehicle_id, duration, incident_type, frame_copy, position):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_pretty = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Obtener nombre de cámara para el log
        try:
            cam_idx = cfg.CAMERA_CHANNELS.index(channel)
            cam_name = cfg.CAMERA_NAMES[cam_idx]
        except:
            cam_name = f"Cam_{channel}"

        # 1. REGISTRAR INCIDENTE EN ESTADÍSTICAS
        self.stats_manager.log_incident(cam_name)

        # 2. PREPARAR EVIDENCIA VISUAL
        cx, cy = position
        # Dibujo sobre la imagen
        cv2.circle(frame_copy, (cx, cy), 40, (0, 0, 255), 3)
        cv2.line(frame_copy, (cx - 50, cy), (cx + 50, cy), (0, 0, 255), 2)
        cv2.line(frame_copy, (cx, cy - 50), (cx, cy + 50), (0, 0, 255), 2)

        label_top = f"INCIDENTE: {incident_type.upper()}"
        label_bot = f"ID: {vehicle_id} | {int(duration)}s DETENIDO"

        cv2.putText(frame_copy, label_top, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame_copy, label_bot, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_copy, timestamp_pretty, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 3. GUARDAR ARCHIVO
        folder = "evidencias"
        os.makedirs(folder, exist_ok=True)

        filename = f"{folder}/{incident_type}_{cam_name}_ID{vehicle_id}_{timestamp_str}.jpg"
        success = cv2.imwrite(filename, frame_copy)

        if success:
            print(f"\n[ALERTA] 📸 Evidencia guardada: {filename}")
        else:
            print(f"\n[ERROR] No se pudo guardar la evidencia.")

    def trigger_alert(self, channel, vehicle_id, duration, incident_type, frame):
        frame_copy = frame.copy()
        pos = self.vehicle_data[channel][vehicle_id]['last_pos']

        t = threading.Thread(
            target=self.handle_incident_log,
            args=(channel, vehicle_id, duration, incident_type, frame_copy, pos)
        )
        t.daemon = True
        t.start()

    def update_vehicle_status(self, channel, tracked_objects, current_light_state, frame_for_evidence):
        current_time = time.time()
        active_ids = []

        for obj in tracked_objects:
            x, y, x2, y2, vid = obj
            cx, cy = (x + x2) // 2, (y + y2) // 2
            active_ids.append(vid)

            if vid not in self.vehicle_data[channel]:
                self.vehicle_data[channel][vid] = {
                    'last_pos': (cx, cy),
                    'accumulated_time': 0.0,
                    'last_update_time': current_time,
                    'incident_type': 'none',
                    'alert_sent': False
                }
            else:
                data = self.vehicle_data[channel][vid]
                prev_x, prev_y = data['last_pos']
                dist = math.hypot(cx - prev_x, cy - prev_y)
                dt = current_time - data['last_update_time']

                # 1. ¿SE MOVIÓ?
                if dist > self.STOP_THRESHOLD:
                    data['accumulated_time'] = 0.0
                    data['incident_type'] = 'none'
                    data['alert_sent'] = False  # Resetear alerta si se mueve
                else:
                    # 2. ESTÁ QUIETO
                    if current_light_state == 'green':
                        data['accumulated_time'] += dt

                        if data['accumulated_time'] > self.ACCIDENT_TIME:
                            # Si aún no tiene tipo, es avería por defecto
                            if data['incident_type'] == 'none':
                                data['incident_type'] = 'breakdown'

                            # Enviar alerta una sola vez
                            if not data['alert_sent']:
                                self.trigger_alert(channel, vid, data['accumulated_time'],
                                                   data['incident_type'], frame_for_evidence)
                                data['alert_sent'] = True
                    else:
                        # Pausa en rojo/amarillo
                        pass

                data['last_pos'] = (cx, cy)
                data['last_update_time'] = current_time

        # Limpieza de IDs viejos
        known_ids = list(self.vehicle_data[channel].keys())
        for vid in known_ids:
            if vid not in active_ids:
                del self.vehicle_data[channel][vid]

    def check_collisions(self, channel):
        stopped_vehicles = []
        for vid, data in self.vehicle_data[channel].items():
            # Solo consideramos vehículos que ya superaron el tiempo de alerta
            if data['accumulated_time'] > self.ACCIDENT_TIME:
                stopped_vehicles.append((vid, data['last_pos']))

        n = len(stopped_vehicles)
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    id1, pos1 = stopped_vehicles[i]
                    id2, pos2 = stopped_vehicles[j]
                    dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])

                    if dist < self.COLLISION_DIST:
                        # Actualizar ambos a COLISION
                        self.vehicle_data[channel][id1]['incident_type'] = 'collision'
                        self.vehicle_data[channel][id2]['incident_type'] = 'collision'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_time = time.time()
            if current_time - self.click_cooldown < 0.3: return
            self.click_cooldown = current_time

            if not self.is_editing:
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
            else:
                if x < 640:
                    self.edit_points.append([x, y])

    # --- MÉTODOS DE CONTROL DE TRÁFICO ---
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
            rects = []
            if len(bboxes) > 0:
                bboxes = (bboxes / 0.5).astype(int)
                rects = bboxes.tolist()

            tracked_objects = self.trackers[channel].update(rects)
            self.last_detections[channel] = tracked_objects

            # --- ACTUALIZACIÓN DE ESTADÍSTICAS DE FLUJO ---
            # El ID más alto del tracker nos dice cuántos coches han pasado
            max_id = self.trackers[channel].id_count
            try:
                cam_idx = cfg.CAMERA_CHANNELS.index(channel)
                cam_name = cfg.CAMERA_NAMES[cam_idx]
                self.stats_manager.update_flow(cam_name, max_id)
            except:
                pass

            current_light = self.traffic_states[channel]
            # Analizar incidentes (Avería/Colisión)
            self.update_vehicle_status(channel, tracked_objects, current_light, frame)
            self.check_collisions(channel)

            # Conteo para semáforo
            current_rect = [self.live_zones[channel]['main']]
            current_arrow = [self.live_zones[channel]['arrow']]

            c_main, c_arrow = 0, 0
            for obj in tracked_objects:
                x, y, x2, y2, _ = obj
                cx, cy = (x + x2) // 2, (y + y2) // 2
                if self.detector.is_valid_detection(cx, cy, current_rect): c_main += 1
                if self.detector.is_valid_detection(cx, cy, current_arrow): c_arrow += 1

            self.detection_counts[channel] = {'main': c_main, 'arrow': c_arrow}

        # --- DIBUJAR ---
        for obj in self.last_detections[channel]:
            x, y, x2, y2, vid = obj
            v_data = self.vehicle_data[channel].get(vid, {})
            accumulated_time = v_data.get('accumulated_time', 0)
            incident_type = v_data.get('incident_type', 'none')

            color = (255, 0, 0)
            label = f"ID:{vid}"

            if incident_type == 'collision':
                color = (255, 0, 255)
                label = f"COLISION {int(accumulated_time)}s"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 4)
                cv2.putText(frame, "POSIBLE COLISION", (x, y - 25), cv2.FONT_HERSHEY_BOLD, 0.6, color, 2)

            elif incident_type == 'breakdown':
                color = (0, 0, 255)
                label = f"AVERIA {int(accumulated_time)}s"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 2)

            elif accumulated_time > 10.0:
                color = (0, 255, 255)
                label = f"ID:{vid} {int(accumulated_time)}s"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 1)
            else:
                cv2.rectangle(frame, (x, y), (x2, y2), color, 1)

            cv2.circle(frame, ((x + x2) // 2, (y + y2) // 2), 3, color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

            # Chequeo periódico de guardado de estadísticas (CSV)
            self.stats_manager.check_periodic_save()

            if self.is_editing and self.edit_channel in self.cameras:
                ret, raw = self.cameras[self.edit_channel].read()
                if ret:
                    edit_frame = vis.draw_edit_mode(raw.copy(), self.edit_points,
                                                    f"EDITANDO: {self.edit_channel}",
                                                    self.edit_zone_type)
                    cv2.imshow(window_name, edit_frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    self.is_editing = False
                elif k == ord('z'):
                    if self.edit_points: self.edit_points.pop()
                elif k == ord('t'):
                    self.edit_zone_type = 'arrow' if self.edit_zone_type == 'main' else 'main'
                elif k == ord('s'):
                    new_zone = np.array(self.edit_points) if len(self.edit_points) > 2 else []
                    self.live_zones[self.edit_channel][self.edit_zone_type] = new_zone
                    self.is_editing = False
            else:
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

                dashboard_info = {
                    'phase_idx': self.current_phase,
                    'active_cams': sum(1 for s in self.camera_status.values() if s == 'active'),
                    'intelligent_cams': sum(1 for m in self.system_mode.values() if m == 'INTELLIGENT')
                }

                # OBTENER DATOS DE ESTADÍSTICAS PARA EL DASHBOARD
                stats_data = self.stats_manager.get_dashboard_data()

                final_view = vis.draw_dashboard(grid, dashboard_info, stats_data)
                cv2.imshow(window_name, final_view)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Guardar estadísticas finales antes de salir
        self.stats_manager.save_snapshot()
        self.running = False
        cv2.destroyAllWindows()
        for cap in self.cameras.values(): cap.release()


if __name__ == '__main__':
    TrafficLightSystem().run()