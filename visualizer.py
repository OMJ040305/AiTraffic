import cv2
import numpy as np
import config as cfg


def draw_traffic_light(frame, state, position):
    """Dibuja el semáforo principal"""
    cv2.circle(frame, position, 20, cfg.TRAFFIC_LIGHT_COLORS[state], -1)
    cv2.circle(frame, position, 20, (255, 255, 255), 2)


def draw_arrow_light(frame, state, position):
    """Dibuja el semáforo de flecha"""
    cv2.circle(frame, position, 15, cfg.TRAFFIC_LIGHT_COLORS[state], -1)
    cv2.circle(frame, position, 15, (255, 255, 255), 2)


def draw_direction_arrow(frame, idx, arrow_state):
    """Dibuja la flecha indicadora de dirección"""
    if idx not in [cfg.ESTE_IDX, cfg.OESTE_IDX]: return

    height, width = frame.shape[:2]
    cx, cy = width - 80, 40

    # Polígono de flecha
    pts = np.array([
        [cx - 10, cy - 10], [cx + 5, cy - 10], [cx + 5, cy - 20],
        [cx + 25, cy],
        [cx + 5, cy + 20], [cx + 5, cy + 10], [cx - 10, cy + 10]
    ], np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(frame, [pts], cfg.TRAFFIC_LIGHT_COLORS[arrow_state])
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)


def add_overlay(frame, channel, name, idx, system_state):
    """
    Dibuja la información sobre el video en Modo Grid (Normal)
    """
    height, width = frame.shape[:2]

    # Info de Cámara
    cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mode_txt = f"Modo: {cfg.SYSTEM_MODES[system_state['mode']]}"
    col = (0, 255, 0) if system_state['status'] == 'active' else (0, 0, 255)
    cv2.putText(frame, mode_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Semáforos
    draw_traffic_light(frame, system_state['traffic_color'], (width - 40, 40))

    if idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
        draw_arrow_light(frame, system_state['arrow_color'], (width - 80, 40))
        draw_direction_arrow(frame, idx, system_state['arrow_color'])

    # Debug Visual (Zonas y Conteos)
    if system_state['mode'] == 'INTELLIGENT' and system_state['counts']:
        # Textos de conteo
        cv2.putText(frame, f"RECTO: {system_state['counts']['main']}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
            cv2.putText(frame, f"FLECHA: {system_state['counts']['arrow']}", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar zonas (las zonas vivas que vienen del main)
        main_z, arrow_z = system_state['zones']

        # Dibujar Main (Azul)
        if len(main_z) > 0:
            pts_m = np.array(main_z, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_m], True, (255, 0, 0), 2)

        # Dibujar Flecha (Verde)
        if len(arrow_z) > 0:
            pts_a = np.array(arrow_z, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_a], True, (0, 255, 0), 2)

    return frame


def draw_edit_mode(frame, points, camera_name, zone_type):
    """
    Dibuja la interfaz de EDICIÓN con menú lateral.
    """
    h, w = frame.shape[:2]
    MENU_W = 250

    # Crear lienzo extendido (Video + Menú)
    canvas = np.zeros((h, w + MENU_W, 3), dtype=np.uint8)
    canvas[:, :w] = frame  # Copiar video a la izquierda
    canvas[:, w:] = (40, 40, 40)  # Fondo gris oscuro a la derecha

    # --- DIBUJAR PUNTOS Y LÍNEAS SOBRE EL VIDEO ---
    if len(points) > 0:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
        for p in points:
            cv2.circle(canvas, (p[0], p[1]), 5, (0, 255, 255), -1)
            # Coordenadas pequeñas
            cv2.putText(canvas, f"{p}", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # --- DIBUJAR MENÚ LATERAL ---
    ui_x = w + 15

    # Título (Aquí estaba el error, cambiado a FONT_HERSHEY_SIMPLEX)
    cv2.putText(canvas, "MODO EDICION", (ui_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, camera_name, (ui_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Tipo de zona
    type_text = "ZONA: RECTA" if zone_type == 'main' else "ZONA: FLECHA"
    type_col = (255, 100, 100) if zone_type == 'main' else (100, 255, 100)
    # Usamos SIMPLEX con grosor 2 para que parezca negrita
    cv2.putText(canvas, type_text, (ui_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, type_col, 2)

    # Instrucciones
    cv2.putText(canvas, "Controles:", (ui_x, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    y_inst = 155
    gap = 25
    instructions = [
        "Click: Poner punto",
        "'T': Cambiar Tipo Zona",
        "'Z': Deshacer punto",
        "'S': GUARDAR Y SALIR",
        "'ESC': Cancelar"
    ]

    for i, inst in enumerate(instructions):
        col = (200, 200, 200)
        if "'S'" in inst: col = (0, 255, 0)
        if "'ESC'" in inst: col = (0, 0, 255)
        cv2.putText(canvas, inst, (ui_x, y_inst + (i * gap)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    return canvas