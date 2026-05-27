import numpy as np

class ConfigCamara:
    def __init__(self):
        # Matriz intrínseca estimada para un sensor de móvil estándar (Moto G / similar)
        # Formato: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        # Estos valores son una aproximación estándar de 35mm para móviles de esa época.
        self.mtx = np.array([
            [800, 0, 640],
            [0, 800, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Coeficientes de distorsión radial y tangencial (k1, k2, p1, p2, k3)
        self.dist = np.array([0.1, -0.05, 0, 0, 0], dtype=np.float32)

# Ahora, en tu motor de análisis, simplemente llamarías a esto:
# config = ConfigCamara()
# imagen_corregida = cv2.undistort(imagen, config.mtx, config.dist)
