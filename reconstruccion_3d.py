import cv2
import numpy as np

class Reconstructor3D:
    def __init__(self, config_camara, tamano_cuadro_mm=5.0):
        """
        Inicializa el motor 3D.
        :param config_camara: Instancia de ConfigCamara con mtx y dist.
        :param tamano_cuadro_mm: Tamaño real del lado de un cuadrado del tablero de ajedrez en mm.
        """
        self.mtx = config_camara.mtx
        self.dist = config_camara.dist
        self.tamano_cuadro = tamano_cuadro_mm
        
        # SIFT sustituye a SURF (es más estándar y libre de patentes en OpenCV actual)
        self.sift = cv2.SIFT_create()

    def calcular_3d(self, img1, img2, pts_tablero1, pts_tablero2, mascara_herida):
        """
        Equivalente a f_calcular_3d.m
        Reconstruye la nube de puntos 3D de la herida.
        """
        # 1. Corregir distorsión de la lente (undistortImage)
        img1_undist = cv2.undistort(img1, self.mtx, self.dist)
        img2_undist = cv2.undistort(img2, self.mtx, self.dist)

        # 2. Generar el modelo matemático del tablero (puntos en el mundo real)
        # Asumimos que pts_tablero1 tiene 54 puntos (9x6) como en tu MATLAB
        objp = np.zeros((9 * 6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * self.tamano_cuadro

        # 3. Calcular la pose de las cámaras (extrinsics)
        # solvePnP nos da la Rotación (rvec) y Traslación (tvec) de cada foto
        ret1, rvec1, tvec1 = cv2.solvePnP(objp, pts_tablero1, self.mtx, self.dist)
        ret2, rvec2, tvec2 = cv2.solvePnP(objp, pts_tablero2, self.mtx, self.dist)

        # Convertir vectores de rotación a matrices 3x3
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)

        # Matriz de Proyección P = K * [R | t] (cameraMatrix en MATLAB)
        P1 = np.dot(self.mtx, np.hstack((R1, tvec1)))
        P2 = np.dot(self.mtx, np.hstack((R2, tvec2)))

        # 4. Encontrar puntos clave SIFT solo DENTRO del contorno de la herida
        # Le pasamos la 'mascara_herida' para no perder tiempo analizando la sábana o la mesa
        kp1, des1 = self.sift.detectAndCompute(img1_undist, mascara_herida)
        kp2, des2 = self.sift.detectAndCompute(img2_undist, mascara_herida)

        # 5. Emparejar características (matchFeatures)
        # Usamos FLANN o BruteForce para encontrar qué píxel de la foto 1 es el mismo en la foto 2
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filtro de Lowe (quedarnos solo con emparejamientos muy seguros)
        pts1_buenas = []
        pts2_buenas = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pts1_buenas.append(kp1[m.queryIdx].pt)
                pts2_buenas.append(kp2[m.trainIdx].pt)

        pts1_buenas = np.float32(pts1_buenas).T
        pts2_buenas = np.float32(pts2_buenas).T

        # 6. TRIANGULACIÓN (triangulate)
        # Calcula la posición X,Y,Z en el espacio cruzando los rayos de ambas cámaras
        puntos_4d = cv2.triangulatePoints(P1, P2, pts1_buenas, pts2_buenas)

        # OpenCV devuelve coordenadas homogéneas (4D). Dividimos por la 4ª coordenada para tener 3D.
        puntos_3d = puntos_4d[:3, :] / puntos_4d[3, :]
        puntos_3d = puntos_3d.T  # Forma (N, 3)

        # 7. Extraer el color real de cada punto para pintarlo en el visor 3D
        colores = []
        for pt in pts1_buenas.T:
            x, y = int(pt[0]), int(pt[1])
            color_bgr = img1_undist[y, x]
            colores.append((color_bgr[2], color_bgr[1], color_bgr[0])) # Guardar en RGB

        return puntos_3d, np.array(colores)