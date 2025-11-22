import cv2
import numpy as np
import config as cfg


def draw_traffic_light(frame, state, position):
    cv2.circle(frame, position, 20, cfg.TRAFFIC_LIGHT_COLORS[state], -1)
    cv2.circle(frame, position, 20, (255, 255, 255), 2)


def draw_arrow_light(frame, state, position):
    cv2.circle(frame, position, 15, cfg.TRAFFIC_LIGHT_COLORS[state], -1)
    cv2.circle(frame, position, 15, (255, 255, 255), 2)


def draw_direction_arrow(frame, idx, arrow_state):
    if idx not in [cfg.ESTE_IDX, cfg.OESTE_IDX]: return

    height, width = frame.shape[:2]
    cx, cy = width - 80, 40

    # Dibujar flecha
    pts = np.array([
        [cx - 10, cy - 10], [cx + 5, cy - 10], [cx + 5, cy - 20],
        [cx + 25, cy],
        [cx + 5, cy + 20], [cx + 5, cy + 10], [cx - 10, cy + 10]
    ], np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(frame, [pts], cfg.TRAFFIC_LIGHT_COLORS[arrow_state])
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)


def add_overlay(frame, channel, name, idx, system_state):
    """
    system_state es un diccionario con:
    mode, status, traffic_color, arrow_color, counts, zones
    """
    height, width = frame.shape[:2]

    # Textos Informativos
    cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    mode_txt = f"Modo: {cfg.SYSTEM_MODES[system_state['mode']]}"
    col = (0, 255, 0) if system_state['status'] == 'active' else (0, 0, 255)
    cv2.putText(frame, mode_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Semáforos
    draw_traffic_light(frame, system_state['traffic_color'], (width - 40, 40))

    if idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
        draw_arrow_light(frame, system_state['arrow_color'], (width - 80, 40))
        draw_direction_arrow(frame, idx, system_state['arrow_color'])

    # Debugging visual (Zonas y conteos)
    if system_state['mode'] == 'INTELLIGENT' and system_state['counts']:
        cv2.putText(frame, f"RECTO: {system_state['counts']['main']}", (10, height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if idx in [cfg.ESTE_IDX, cfg.OESTE_IDX]:
            cv2.putText(frame, f"FLECHA: {system_state['counts']['arrow']}", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dibujar zonas
        main_z, arrow_z = system_state['zones']
        for z in main_z: cv2.polylines(frame, [z], True, (255, 0, 0), 2)
        for z in arrow_z: cv2.polylines(frame, [z], True, (0, 255, 0), 2)

    return frame