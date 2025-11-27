import math


class EuclideanDistTracker:
    def __init__(self):
        # Almacena las posiciones centrales de los objetos: {id: (x, y)}
        self.center_points = {}
        # Contador para asignar nuevos IDs únicos
        self.id_count = 0

    def update(self, objects_rect):
        """
        Calcula la distancia euclidiana entre los nuevos objetos detectados
        y los objetos previos para asignarles el mismo ID.

        Args:
            objects_rect: Lista de cajas [x, y, w, h] detectadas en el frame actual.

        Returns:
            Lista de cajas con ID [x, y, w, h, id]
        """
        objects_bbs_ids = []

        # Obtener el punto central de cada nuevo objeto detectado
        for rect in objects_rect:
            x, y, x2, y2 = rect
            cx = (x + x2) // 2
            cy = (y + y2) // 2

            # Intentar coincidir con objetos existentes
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # TOLERANCIA AUMENTADA: 100px
                # Si el centro del objeto está a menos de 100px del anterior, es el mismo.
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, x2, y2, id])
                    same_object_detected = True
                    break

            # Si no se encuentra coincidencia, es un auto nuevo
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, x2, y2, self.id_count])
                self.id_count += 1

        # Limpiar IDs que ya no están en pantalla para liberar memoria
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        self.center_points = new_center_points.copy()
        return objects_bbs_ids
