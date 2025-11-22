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

    pts = np.array([
        [cx - 10, cy - 10], [cx + 5, cy - 10], [cx + 5, cy - 20],
        [cx + 25, cy],
        [cx + 5, cy + 20], [cx + 5, cy + 10], [cx - 10, cy + 10]
    ], np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(frame, [pts], cfg.TRAFFIC_LIGHT_COLORS[arrow_state])
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)


def add_overlay(frame, channel, name, idx, system_state):
    """Dibuja la información sobre cada cámara individual"""
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
        cv2.putText(frame, f"RECTO: {system_state['counts']['main']}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
            cv2.putText(frame, f"FLECHA: {system_state['counts']['arrow']}", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        main_z, arrow_z = system_state['zones']

        if len(main_z) > 0:
            pts_m = np.array(main_z, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_m], True, (255, 0, 0), 2)
        if len(arrow_z) > 0:
            pts_a = np.array(arrow_z, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_a], True, (0, 255, 0), 2)

    return frame


def draw_edit_mode(frame, points, camera_name, zone_type):
    """Dibuja la interfaz de EDICIÓN con menú lateral."""
    h, w = frame.shape[:2]
    MENU_W = 300

    canvas = np.zeros((h, w + MENU_W, 3), dtype=np.uint8)
    canvas[:, :w] = frame
    canvas[:, w:] = (40, 40, 40)  # Fondo gris

    # Dibujo sobre video
    if len(points) > 0:
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
        for p in points:
            cv2.circle(canvas, (p[0], p[1]), 5, (0, 255, 255), -1)
            cv2.putText(canvas, f"{p}", (p[0] + 10, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Menú Lateral
    ui_x = w + 20
    cv2.putText(canvas, "MODO EDICION", (ui_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, camera_name, (ui_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    type_text = "ZONA: RECTA" if zone_type == 'main' else "ZONA: FLECHA"
    type_col = (255, 100, 100) if zone_type == 'main' else (100, 255, 100)
    cv2.putText(canvas, type_text, (ui_x, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, type_col, 2)

    # Instrucciones
    instrucciones = [
        ("CONTROLES:", (255, 255, 0)),
        ("Click: Poner punto", (200, 200, 200)),
        ("'T': Cambiar Tipo Zona", (200, 200, 200)),
        ("'Z': Deshacer punto", (200, 200, 200)),
        ("'S': GUARDAR CAMBIOS", (0, 255, 0)),
        ("'ESC': Cancelar", (0, 0, 255))
    ]

    y_start = 180
    for text, color in instrucciones:
        cv2.putText(canvas, text, (ui_x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_start += 30

    return canvas


def draw_dashboard(grid_frame, info_data):
    """
    Dibuja el menú lateral principal (Dashboard) junto al grid de cámaras.
    info_data: diccionario con {phase_idx, active_cams, intelligent_cams}
    """
    h, w = grid_frame.shape[:2]
    MENU_W = 300  # Ancho del menú lateral

    # Crear lienzo grande
    canvas = np.zeros((h, w + MENU_W, 3), dtype=np.uint8)
    canvas[:, :w] = grid_frame
    canvas[:, w:] = (30, 30, 30)  # Gris oscuro (un poco más oscuro que el editor)

    ui_x = w + 20

    # --- TÍTULO ---
    cv2.putText(canvas, "CONTROL DE", (ui_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "TRAFICO AI", (ui_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(canvas, (ui_x, 110), (w + MENU_W - 20, 110), (100, 100, 100), 1)

    # --- ESTADO DEL SISTEMA ---
    cv2.putText(canvas, "ESTADO SISTEMA", (ui_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.putText(canvas, f"Camaras Activas: {info_data['active_cams']}/4", (ui_x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1)
    cv2.putText(canvas, f"Modo Inteligente: {info_data['intelligent_cams']}/4", (ui_x, 210), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)

    cv2.line(canvas, (ui_x, 240), (w + MENU_W - 20, 240), (100, 100, 100), 1)

    # --- FASE ACTUAL (SEMÁFORO) ---
    cv2.putText(canvas, "FASE ACTUAL", (ui_x, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Lista de fases para resaltar la activa
    phases = ["1. Flechas E-O", "2. Rectos E-O", "3. Norte", "4. Sur"]
    current = info_data['phase_idx']

    y_ph = 320
    for i, ph_name in enumerate(phases):
        color = (50, 50, 50)  # Color apagado
        thickness = 1
        prefix = "  "

        if i == current:
            color = (0, 255, 0)  # Verde brillante
            thickness = 2
            prefix = "> "

        cv2.putText(canvas, prefix + ph_name, (ui_x, y_ph), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        y_ph += 35

    cv2.line(canvas, (ui_x, 480), (w + MENU_W - 20, 480), (100, 100, 100), 1)

    # --- INSTRUCCIONES ---
    cv2.putText(canvas, "ACCIONES", (ui_x, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, "Click en camara:", (ui_x, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(canvas, "para EDITAR ZONA", (ui_x, 575), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.rectangle(canvas, (ui_x, 650), (ui_x + 100, 690), (0, 0, 200), -1)
    cv2.putText(canvas, "Q: SALIR", (ui_x + 10, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return canvas