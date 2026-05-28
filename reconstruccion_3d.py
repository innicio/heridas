import cv2
import numpy as np
from scipy.spatial import KDTree

class Reconstructor3D:
    def __init__(self, config_camara):
        self.config = config_camara
        # Inicializamos el extractor SIFT (Scale-Invariant Feature Transform)
        self.sift = cv2.SIFT_create()
        
        # Configuramos el emparejador FLANN (Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def filtrar_ruido_estadistico(self, nube_puntos, k_vecinos=15, factor_alpha=1.0):
        """
        Filtro SOR (Statistical Outlier Removal).
        Elimina puntos flotantes comparando la densidad local con la media global.
        """
        if nube_puntos is None or len(nube_puntos) < k_vecinos:
            return nube_puntos
            
        tree = KDTree(nube_puntos)
        distancias, _ = tree.query(nube_puntos, k=k_vecinos + 1)
        distancias_vecinos = distancias[:, 1:]
        
        medias_locales = np.mean(distancias_vecinos, axis=1)
        mu = np.mean(medias_locales)
        sigma = np.std(medias_locales)
        
        umbral_corte = mu + (factor_alpha * sigma)
        mascara_inliers = medias_locales <= umbral_corte
        
        return nube_puntos[mascara_inliers]

    def calcular_3d(self, lista_imagenes, lista_ref_puntos, mascara_roi):
        """
        Procesa una secuencia de imágenes (SFM) para reconstruir la nube de puntos 3D.
        """
        if not lista_imagenes or len(lista_imagenes) < 2:
            return None, None

        nube_total = []
        num_vistas = len(lista_imagenes)
        
        # Usamos una matriz de cámara genérica si la configuración no la provee
        # (Idealmente, tu ConfigCamara debería devolver los valores calibrados reales)
        h, w = lista_imagenes[0].shape[:2]
        focal_length = w  # Aproximación estándar
        centro = (w / 2, h / 2)
        matriz_camara = np.array([
            [focal_length, 0, centro[0]],
            [0, focal_length, centro[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        # Iteramos sobre pares consecutivos de la secuencia
        for idx in range(num_vistas - 1):
            img1 = lista_imagenes[idx]
            img2 = lista_imagenes[idx + 1]

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # 1. Extracción de Keypoints limitados a la zona de la úlcera (mascara_roi)
            kp1, des1 = self.sift.detectAndCompute(gray1, mask=mascara_roi)
            kp2, des2 = self.sift.detectAndCompute(gray2, mask=mascara_roi)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                continue

            # 2. Emparejamiento rápido y filtrado de Lowe
            matches = self.flann.knnMatch(des1, des2, k=2)
            puntos_buenos_1 = []
            puntos_buenos_2 = []

            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    puntos_buenos_1.append(kp1[m.queryIdx].pt)
                    puntos_buenos_2.append(kp2[m.trainIdx].pt)

            pts1 = np.float32(puntos_buenos_1)
            pts2 = np.float32(puntos_buenos_2)

            if len(pts1) < 8:
                continue

            # 3. Geometría Epipolar (Filtrado RANSAC de artefactos visuales)
            E, mascara_ransac = cv2.findEssentialMat(pts1, pts2, matriz_camara, cv2.RANSAC, 0.999, 1.0)
            if E is None:
                continue

            pts1_inliers = pts1[mascara_ransac.ravel() == 1]
            pts2_inliers = pts2[mascara_ransac.ravel() == 1]

            # 4. Recuperación de la pose de la cámara (Rotación y Traslación)
            _, R, t, mascara_pose = cv2.recoverPose(E, pts1_inliers, pts2_inliers, matriz_camara)

            # 5. Triangulación Espacial (El cálculo del relieve Z)
            # Matriz de proyección de la cámara 1 (Origen)
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = matriz_camara @ P1
            
            # Matriz de proyección de la cámara 2 (Desplazada)
            P2 = np.hstack((R, t))
            P2 = matriz_camara @ P2

            puntos_4d_homogeneos = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
            
            # Convertimos coordenadas homogéneas (4D) a cartesianas (3D)
            puntos_3d = puntos_4d_homogeneos[:3, :] / puntos_4d_homogeneos[3, :]
            puntos_3d = puntos_3d.T # Transponemos para tener formato (N, 3)

            # Solo guardamos los puntos que están "delante" de la cámara (Z > 0)
            puntos_validos = puntos_3d[puntos_3d[:, 2] > 0]
            
            if len(puntos_validos) > 0:
                nube_total.append(puntos_validos)

        # 6. Consolidación y filtrado estadístico final
        if len(nube_total) > 0:
            nube_consolidada = np.vstack(nube_total)
            nube_limpia = self.filtrar_ruido_estadistico(nube_consolidada, k_vecinos=20, factor_alpha=1.2)
            return nube_limpia, None
            
        return None, None