import csv
import time
import datetime
import os
import config as cfg


class StatsManager:
    def __init__(self):
        self.filename = "registro_trafico.csv"
        self.initialize_csv()

        # Contadores en memoria para el Dashboard (Visualización en vivo)
        # Estructura: { 'Camara Norte': 0, ... }
        self.vehicle_counts = {name: 0 for name in cfg.CAMERA_NAMES}
        self.incident_counts = {name: 0 for name in cfg.CAMERA_NAMES}

        # Control de tiempo para guardado periódico (cada minuto)
        self.last_save_time = time.time()
        self.save_interval = 60.0

    def initialize_csv(self):
        """Crea el archivo con encabezados si no existe"""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Encabezados del Excel
                headers = ["Timestamp", "Fecha", "Hora", "Camara", "Vehiculos_Totales", "Nuevos_Incidentes"]
                writer.writerow(headers)

    def update_flow(self, camera_name, total_tracker_id):
        """
        Actualiza el conteo basado en el ID máximo del tracker.
        Si el tracker va en el ID 500, han pasado 500 coches.
        """
        # Solo actualizamos si el número crece (para evitar errores al reiniciar trackers)
        if total_tracker_id > self.vehicle_counts[camera_name]:
            self.vehicle_counts[camera_name] = total_tracker_id

    def log_incident(self, camera_name):
        """Registra un nuevo incidente"""
        self.incident_counts[camera_name] += 1
        self.save_snapshot(force=True)  # Guardar inmediatamente si hay accidente

    def check_periodic_save(self):
        """Guarda los datos en el CSV cada minuto"""
        if time.time() - self.last_save_time > self.save_interval:
            self.save_snapshot()
            self.last_save_time = time.time()

    def save_snapshot(self, force=False):
        """Escribe el estado actual en el CSV"""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        try:
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for cam_name in cfg.CAMERA_NAMES:
                    # Escribimos una fila por cada cámara
                    writer.writerow([
                        timestamp,
                        date_str,
                        time_str,
                        cam_name,
                        self.vehicle_counts[cam_name],
                        self.incident_counts[cam_name]
                    ])
            if not force:
                print(f"[STATS] 💾 Datos guardados en {self.filename}")
        except Exception as e:
            print(f"[STATS] Error guardando CSV: {e}")

    def get_dashboard_data(self):
        """Retorna datos para pintar en el visualizador"""
        total_cars = sum(self.vehicle_counts.values())
        total_incidents = sum(self.incident_counts.values())
        return self.vehicle_counts, total_cars, total_incidents