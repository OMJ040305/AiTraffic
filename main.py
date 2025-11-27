import datetime
import math
import os
import threading
import time

import cv2
import numpy as np
import requests

# Importar módulos propios
import config as cfg
import visualizer as vis
from detector import VehicleDetector
from stats import StatsManager
from tracker import EuclideanDistTracker


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
        self.vehicle_data = {ch: {} for ch in cfg.CAMERA_CHANNELS}

        # --- GESTOR DE ESTADÍSTICAS ---
        self.stats_manager = StatsManager()

        # --- CONFIGURACIÓN DE INCIDENTES ---
        self.STOP_THRESHOLD = 15

        # ==========================================
        # ⚠️ CONFIGURACIÓN DE TIEMPO
        self.ACCIDENT_TIME = 20.0  # Segundos detenido en VERDE para considerar avería
        # ==========================================

        self.COLLISION_DIST = 150

        # Caché visual
        self.last_detections = {ch: [] for ch in cfg.CAMERA_CHANNELS}
        self.frame_counter = 0

        # OPTIMIZACIÓN
        self.detection_interval = 5

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

    # --- GESTIÓN DE EVIDENCIAS Y WEBHOOK (CON IMAGEN) ---
    def handle_incident_log(self, channel, vehicle_id, duration, incident_type, frame_copy, position):
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamp_pretty = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            cam_idx = cfg.CAMERA_CHANNELS.index(channel)
            cam_name = cfg.CAMERA_NAMES[cam_idx]
        except:
            cam_name = f"Cam_{channel}"

        # Registrar estadísticas localmente
        self.stats_manager.log_incident(cam_name)

        # Dibujar sobre la evidencia visual
        cx, cy = position
        cv2.circle(frame_copy, (cx, cy), 40, (0, 0, 255), 3)
        label_top = f"INCIDENTE: {incident_type.upper()}"
        label_bot = f"ID: {vehicle_id} | {int(duration)}s DETENIDO"

        cv2.putText(frame_copy, label_top, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame_copy, label_bot, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_copy, timestamp_pretty, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Guardar imagen en disco
        folder = "evidencias"
        os.makedirs(folder, exist_ok=True)
        filename = f"{folder}/{incident_type}_{cam_name}_ID{vehicle_id}_{timestamp_str}.jpg"
        cv2.imwrite(filename, frame_copy)
        print(f"[ALERTA] 📸 Evidencia guardada en disco: {filename}")

        # ---------------------------------------------------------
        # ENVÍO AL WEBHOOK DE N8N (DATOS + IMAGEN)
        # ---------------------------------------------------------
        webhook_url = "https://n8n.trazo.xyz/webhook/sendMessageImg"

        msj_intro = "⚠️ ALERTA DE TRAFICO"
        if incident_type == 'breakdown':
            msj_intro = "⚠️ POSIBLE VEHICULO AVERIADO"
        elif incident_type == 'collision':
            msj_intro = "💥 POSIBLE CHOQUE"

        # Datos de texto
        payload = {
            "tipo_evento": "ALERTA_TRAFICO",
            "camara": cam_name,
            "tipo_incidente": incident_type,
            "id_vehiculo": str(vehicle_id),
            "duracion_detenido": f"{int(duration)} segundos",
            "fecha_hora": timestamp_pretty,
            "mensaje": f"{msj_intro} en {cam_name}. Vehiculo ID {vehicle_id} detenido {int(duration)}s en semaforo VERDE."
        }

        print(f"📡 Enviando FOTO y datos a N8N para ID {vehicle_id}...")

        try:
            # Abrimos el archivo en modo lectura binaria ('rb')
            with open(filename, 'rb') as img_file:
                # Diccionario de archivos para la petición multipart
                # 'imagen_evidencia': es el nombre del campo que verás en n8n (Binary property)
                files_data = {
                    'imagen_evidencia': (os.path.basename(filename), img_file, 'image/jpeg')
                }

                # Enviamos POST multipart/form-data
                # Nota: usamos 'data' para el JSON plano y 'files' para el archivo
                response = requests.post(webhook_url, data=payload, files=files_data, timeout=15)

            if response.status_code == 200:
                print(f"[N8N] ✅ Alerta e imagen enviadas correctamente.")
            else:
                print(f"[N8N] ⚠️ Error al enviar. Código: {response.status_code} | Respuesta: {response.text}")

        except Exception as e:
            print(f"[N8N] ❌ Error de conexión al subir imagen: {e}")
        # ---------------------------------------------------------

    def trigger_alert(self, channel, vehicle_id, duration, incident_type, frame):
        frame_copy = frame.copy()
        try:
            pos = self.vehicle_data[channel][vehicle_id]['last_pos']
            t = threading.Thread(target=self.handle_incident_log,
                                 args=(channel, vehicle_id, duration, incident_type, frame_copy, pos))
            t.daemon = True
            t.start()
        except KeyError:
            pass

    def update_vehicle_status(self, channel, tracked_objects, main_light, arrow_light, main_zone, arrow_zone,
                              frame_for_evidence):
        current_time = time.time()
        active_ids = []

        for obj in tracked_objects:
            x, y, x2, y2, vid = obj
            cx, cy = (x + x2) // 2, (y + y2) // 2
            active_ids.append(vid)

            if vid not in self.vehicle_data[channel]:
                self.vehicle_data[channel][vid] = {
                    'last_pos': (cx, cy), 'accumulated_time': 0.0, 'last_update_time': current_time,
                    'incident_type': 'none', 'alert_sent': False, 'lane_type': 'unknown'
                }
            else:
                data = self.vehicle_data[channel][vid]
                dist = math.hypot(cx - data['last_pos'][0], cy - data['last_pos'][1])
                dt = current_time - data['last_update_time']

                # Detectar carril
                is_in_arrow = self.detector.is_valid_detection(cx, cy, [arrow_zone])
                is_in_main = self.detector.is_valid_detection(cx, cy, [main_zone])

                relevant_light_color = 'red'
                if is_in_arrow:
                    relevant_light_color = arrow_light
                    data['lane_type'] = 'arrow'
                elif is_in_main:
                    relevant_light_color = main_light
                    data['lane_type'] = 'main'
                else:
                    relevant_light_color = main_light
                    data['lane_type'] = 'main'

                # --- MODIFICACIÓN DE LA LÓGICA DE AVERÍA (PAUSAR EN ROJO) ---
                if dist > self.STOP_THRESHOLD:
                    # El auto se mueve: Reiniciar contadores (está circulando bien)
                    data['accumulated_time'] = 0.0
                    data['incident_type'] = 'none'
                    data['alert_sent'] = False
                else:
                    # El auto está detenido (distancia < threshold)
                    if relevant_light_color == 'green':
                        # Solo si es VERDE sumamos tiempo para detectar avería
                        data['accumulated_time'] += dt

                        if data['accumulated_time'] > self.ACCIDENT_TIME:
                            if data['incident_type'] == 'none':
                                data['incident_type'] = 'breakdown'

                            if not data['alert_sent']:
                                self.trigger_alert(channel, vid, data['accumulated_time'], data['incident_type'],
                                                   frame_for_evidence)
                                data['alert_sent'] = True
                    else:
                        # Si es ROJO o AMARILLO:
                        # NO REINICIAMOS. Simplemente "pausamos" (no hacemos nada).
                        # El acumulado se mantiene igual hasta que vuelva a ponerse en verde.
                        pass

                data['last_pos'] = (cx, cy)
                data['last_update_time'] = current_time

        known = list(self.vehicle_data[channel].keys())
        for vid in known:
            if vid not in active_ids: del self.vehicle_data[channel][vid]

    def check_collisions(self, channel):
        stopped = []
        for vid, data in self.vehicle_data[channel].items():
            if data['accumulated_time'] > self.ACCIDENT_TIME:
                stopped.append((vid, data['last_pos']))

        n = len(stopped)
        if n >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    id1, p1 = stopped[i]
                    id2, p2 = stopped[j]
                    if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < self.COLLISION_DIST:
                        self.vehicle_data[channel][id1]['incident_type'] = 'collision'
                        self.vehicle_data[channel][id2]['incident_type'] = 'collision'

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if time.time() - self.click_cooldown < 0.3: return
            self.click_cooldown = time.time()
            if not self.is_editing:
                if x > 960: return
                col, row = (0 if x < 480 else 1), (0 if y < 360 else 1)
                idx = row * 2 + col
                if idx < len(cfg.CAMERA_CHANNELS):
                    self.edit_channel = cfg.CAMERA_CHANNELS[idx]
                    self.edit_points = []
                    self.is_editing = True
                    self.edit_zone_type = 'main'
            else:
                if x < 640: self.edit_points.append([x, y])

    # --- CONTROL DE TRÁFICO ---
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
        for ch in cfg.CAMERA_CHANNELS: self.traffic_states[ch] = 'red'; self.arrow_states[ch] = 'red'
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
                if elapsed >= cfg.PHASE_TIMES[self.current_phase] - cfg.YELLOW_TIME:
                    if elapsed < cfg.PHASE_TIMES[self.current_phase]:
                        self.set_lights(self.current_phase, 'yellow')
                    else:
                        next_ph = (self.current_phase + 1) % 4
                        skipped = 0
                        while self.should_skip_phase(next_ph) and skipped < 4:
                            print(f"⏭️ Saltando fase {next_ph} (Sin vehiculos)")
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
            scale_factor = 0.4
            small = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)))

            bboxes = self.detector.detect(small)
            rects = (bboxes / scale_factor).astype(int).tolist() if len(bboxes) > 0 else []

            tracked_objects = self.trackers[channel].update(rects)
            self.last_detections[channel] = tracked_objects

            try:
                cam_idx = cfg.CAMERA_CHANNELS.index(channel)
                cam_name = cfg.CAMERA_NAMES[cam_idx]
                self.stats_manager.update_flow(cam_name, self.trackers[channel].id_count)
            except:
                pass

            curr_main_light = self.traffic_states[channel]
            curr_arrow_light = self.arrow_states[channel]
            main_zone = self.live_zones[channel]['main']
            arrow_zone = self.live_zones[channel]['arrow']

            self.update_vehicle_status(channel, tracked_objects, curr_main_light, curr_arrow_light,
                                       main_zone, arrow_zone, frame)

            self.check_collisions(channel)

            mz_list = [main_zone]
            az_list = [arrow_zone]
            cm, ca = 0, 0
            for obj in tracked_objects:
                cx, cy = (obj[0] + obj[2]) // 2, (obj[1] + obj[3]) // 2
                if self.detector.is_valid_detection(cx, cy, mz_list): cm += 1
                if self.detector.is_valid_detection(cx, cy, az_list): ca += 1
            self.detection_counts[channel] = {'main': cm, 'arrow': ca}

        for obj in self.last_detections[channel]:
            x, y, x2, y2, vid = obj
            v_data = self.vehicle_data[channel].get(vid, {})
            accum = v_data.get('accumulated_time', 0)
            itype = v_data.get('incident_type', 'none')

            if itype == 'breakdown' or accum > self.ACCIDENT_TIME:
                color = (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x2, y2), color, 4)
                mins = int(accum // 60)
                secs = int(accum % 60)
                label = f"ALERTA {mins:02d}:{secs:02d}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x, y - 35), (x + w_text + 10, y), color, -1)
                cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif itype == 'collision':
                color = (255, 0, 255)
                label = f"CHOQUE {int(accum)}s"
                cv2.rectangle(frame, (x, y), (x2, y2), color, 4)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                color = (255, 0, 0)
                if accum > 10.0:
                    color = (0, 255, 255)
                    label = f"ID:{vid} | {int(accum)}s"
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x2, y2), color, 1)
                    cv2.putText(frame, f"ID:{vid}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return frame

    def monitor_cameras(self):
        while self.running:
            now = time.time()
            for ch in cfg.CAMERA_CHANNELS:
                if self.camera_status[ch] == 'active' and (now - self.last_frame_time[ch] > cfg.CAMERA_TIMEOUT):
                    print(f"⚠️ WATCHDOG: Timeout en camara {ch}. Cambiando a STANDARD.")
                    self.camera_status[ch] = 'failed'
                    self.system_mode[ch] = 'STANDARD'
                    self.attempt_reconnect(ch)
            time.sleep(2)

    def attempt_reconnect(self, channel):
        try:
            if channel in self.cameras: self.cameras[channel].release()
            cap = cv2.VideoCapture(channel)
            if cap.isOpened():
                self.cameras[channel] = cap
                self.camera_status[channel] = 'active'
                self.last_frame_time[channel] = time.time()
        except:
            pass

    def initialize_cameras(self):
        print("\n" + "=" * 50)
        print("   INICIANDO SECUENCIA DE CONEXION DE CAMARAS")
        print("=" * 50)
        for i, ch in enumerate(cfg.CAMERA_CHANNELS):
            cam_name = cfg.CAMERA_NAMES[i]
            print(f"\n[..] Conectando {cam_name} (Input: {ch})...")
            try:
                cap = cv2.VideoCapture(ch)
                time.sleep(1.5)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.cameras[ch] = cap
                        self.camera_status[ch] = 'active'
                        self.system_mode[ch] = 'INTELLIGENT'
                        self.last_frame_time[ch] = time.time()
                        print(f"✅ EXITO: {cam_name} conectada.")
                    else:
                        print(f"⚠️ {cam_name} devolvió imagen vacía.")
                        self.camera_status[ch] = 'failed'
                        self.system_mode[ch] = 'STANDARD'
                        cap.release()
                else:
                    print(f"❌ ERROR: No se pudo abrir {cam_name}.")
                    self.camera_status[ch] = 'failed'
                    self.system_mode[ch] = 'STANDARD'
            except Exception as e:
                print(f"❌ ERROR CRITICO en {cam_name}: {e}")
                self.camera_status[ch] = 'failed'
                self.system_mode[ch] = 'STANDARD'
        print("\n" + "=" * 50)
        print(
            f"   RESUMEN: {sum(1 for s in self.camera_status.values() if s == 'active')}/{len(cfg.CAMERA_CHANNELS)} Camaras operativas")
        print("=" * 50 + "\n")

    def run(self):
        print("=== SISTEMA DE TRAFICO AI INICIADO ===")
        self.initialize_cameras()
        window_name = 'Sistema Semaforo'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            self.frame_counter += 1
            self.stats_manager.check_periodic_save()

            if self.is_editing and self.edit_channel in self.cameras:
                ret, raw = self.cameras[self.edit_channel].read()
                if ret:
                    edit_frame = vis.draw_edit_mode(raw.copy(), self.edit_points, f"EDITANDO: {self.edit_channel}",
                                                    self.edit_zone_type)
                    cv2.imshow(window_name, edit_frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    self.is_editing = False
                elif k == ord('z'):
                    if self.edit_points: self.edit_points.pop()
                elif k == ord('s'):
                    self.live_zones[self.edit_channel][self.edit_zone_type] = np.array(self.edit_points)
                    self.is_editing = False
            else:
                frames_list = []
                for i, ch in enumerate(cfg.CAMERA_CHANNELS):
                    frame = np.zeros((360, 480, 3), dtype=np.uint8)
                    camera_ok = False
                    if ch in self.cameras and self.cameras[ch].isOpened():
                        ret, raw = self.cameras[ch].read()
                        if ret:
                            frame = self.process_camera(ch, raw)
                            camera_ok = True
                        else:
                            self.camera_status[ch] = 'failed'
                            self.system_mode[ch] = 'STANDARD'
                    state = {
                        'mode': self.system_mode[ch],
                        'status': self.camera_status[ch],
                        'traffic_color': self.traffic_states[ch],
                        'arrow_color': self.arrow_states[ch],
                        'counts': self.detection_counts[ch],
                        'zones': (self.live_zones[ch]['main'], self.live_zones[ch]['arrow'])
                    }
                    frame = vis.add_overlay(frame, ch, cfg.CAMERA_NAMES[i], i, state)
                    if not camera_ok:
                        cv2.putText(frame, "SIN SENAL", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames_list.append(cv2.resize(frame, (480, 360)))

                top = np.hstack([frames_list[0], frames_list[1]])
                bot = np.hstack([frames_list[2], frames_list[3]])
                grid = np.vstack([top, bot])
                dashboard_info = {'phase_idx': self.current_phase,
                                  'active_cams': sum(1 for s in self.camera_status.values() if s == 'active'),
                                  'intelligent_cams': sum(1 for m in self.system_mode.values() if m == 'INTELLIGENT')}
                stats_data = self.stats_manager.get_dashboard_data()
                final_view = vis.draw_dashboard(grid, dashboard_info, stats_data)
                cv2.imshow(window_name, final_view)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.stats_manager.save_snapshot()
        self.running = False
        cv2.destroyAllWindows()
        for cap in self.cameras.values(): cap.release()


if __name__ == '__main__':
    TrafficLightSystem().run()