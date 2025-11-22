import numpy as np

# --- CONFIGURACIÓN DE CÁMARAS ---
# Canales: Norte, Sur, Este, Oeste
CAMERA_CHANNELS = [0, 4, 3, 2]
CAMERA_NAMES = ["Camara Norte", "Camara Sur", "Camara Este", "Camara Oeste"]

# Índices
NORTE_IDX = 0
SUR_IDX = 1
ESTE_IDX = 2
OESTE_IDX = 3

# --- ZONAS DE DETECCIÓN ---
zonaRectoCamaraEste = np.array([[15, 463], [394, 461], [344, 88], [162, 87]])
zonaFlechaCamaraEste = np.array([[345, 91], [400, 463], [636, 438], [636, 308], [570, 303], [514, 273], [447, 129]])

zonaRectoCamaraOeste = np.array([[0, 475], [392, 479], [362, 29], [148, 35], [2, 338]])
zonaFlechaCamaraOeste = np.array([[393, 475], [635, 475], [638, 327], [574, 305], [542, 275], [470, 78], [362, 35]])

zonaCamaraNorte = np.array([[241, 150], [322, 150], [344, 289], [225, 293]])
zonaCamaraSur = np.array([[237, 207], [205, 347], [316, 344], [306, 209]])

# --- CONFIGURACIÓN DE SEMÁFORO ---
TRAFFIC_LIGHT_COLORS = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'green': (0, 255, 0)
}

SYSTEM_MODES = {
    'INTELLIGENT': 'Inteligente',
    'STANDARD': 'Estandar',
    'FALLBACK': 'Respaldo'
}

# Tiempos (segundos)
PHASE_TIMES = {
    0: 15,  # Flechas E-O
    1: 25,  # Rectos E-O
    2: 20,  # Norte
    3: 20   # Sur
}
YELLOW_TIME = 3
CAMERA_TIMEOUT = 5.0
MAX_FAILURES = 3

# Función helper para obtener zonas según el canal
def get_zones(channel_idx):
    if channel_idx == NORTE_IDX: return [zonaCamaraNorte], []
    elif channel_idx == SUR_IDX: return [zonaCamaraSur], []
    elif channel_idx == ESTE_IDX: return [zonaRectoCamaraEste], [zonaFlechaCamaraEste]
    elif channel_idx == OESTE_IDX: return [zonaRectoCamaraOeste], [zonaFlechaCamaraOeste]
    return [], []