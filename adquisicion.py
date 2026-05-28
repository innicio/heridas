import cv2
import numpy as np

class MotorAnalisisHeridas:
    def __init__(self, chessboard_size=(9, 6)):
        """
        Inicializa el motor de análisis.
        :param chessboard_size: Tupla con el número de esquinas internas del tablero de ajedrez.
                                Por defecto (9, 6) que equivale a los 54 puntos de tu script original.
        """
        self.chessboard_size = chessboard_size
        
        # Cargar aquí los parámetros intrínsecos de la cámara si los tienes exportados
        # Equivalente a 'load datosCamaraMotoG.mat'
        self.camera_matrix = None 
        self.dist_coeffs = None

    def extraer_secuencia_estereo(self, ruta_video, num_imagenes=5, gap_estereo=10):
        """
        Evolución del algoritmo original para capturar múltiples ángulos (SFM).
        Busca secuencialmente fotogramas garantizando el baseline.
        
        Devuelve:
            list: Lista de imágenes capturadas (matrices BGR).
            list: Lista de puntos de referencia refinados (esquinas del tablero).
        """
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise ValueError(f"Error: No se pudo abrir el vídeo '{ruta_video}'")

        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Variables de control
        i = 70       # Fotograma de inicio
        salto = 2    # Avance en la búsqueda si no hay patrón
        
        imagenes_capturadas = []
        puntos_referencia = []

        while i <= frames_total and len(imagenes_capturadas) < num_imagenes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                break

            # Búsqueda del patrón de calibración
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            encontrado, esquinas = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            # Validamos que se detecten exactamente los 54 vértices internos
            if encontrado and esquinas is not None and len(esquinas) == 54:
                
                # Refinamiento subpíxel de las coordenadas del tablero
                criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                esquinas_refinadas = cv2.cornerSubPix(gray, esquinas, (11, 11), (-1, -1), criterios)

                # Almacenamos el fotograma y sus anclas topográficas
                imagenes_capturadas.append(frame)
                puntos_referencia.append(esquinas_refinadas)

                # PUNTO CRÍTICO: Una vez capturado un frame válido, forzamos el salto 
                # del baseline (gap_estereo) para asegurar que la cámara se ha movido
                i += gap_estereo
            else:
                # Si no detecta el patrón completo, avanzamos el salto corto estándar
                i += salto

        cap.release()

        # Validación clínica de los datos
        if len(imagenes_capturadas) < 2:
            raise RuntimeError("La geometría de captura falló: No se encontraron suficientes fotogramas válidos separados por el baseline.")

        return imagenes_capturadas, puntos_referencia

# --- Ejemplo de uso ---
# motor = MotorAnalisisHeridas(chessboard_size=(9, 6))
# img1, img2, pts1, pts2 = motor.escoger_2_imagenes_desde_video("ruta_al_video.mp4")