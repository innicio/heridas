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

    def escoger_2_imagenes_desde_video(self, video_path):
        """
        Analiza un vídeo y extrae dos fotogramas espaciados en el tiempo 
        donde el tablero de calibración sea visible.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Parámetros equivalentes a tu script de MATLAB
        i = 70  # Fotograma de inicio
        salto = 2
        
        imagen1 = None
        imagen2 = None
        refPoints1 = None
        refPoints2 = None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        while i <= total_frames - salto:
            ret, frame = cap.read()
            if not ret:
                break
                
            # OpenCV prefiere trabajar en escala de grises para la detección del tablero
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Buscamos las esquinas del tablero de ajedrez
            ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret_corners:  # Si encuentra los 54 puntos (size1 == 54 en MATLAB)
                imagen1 = frame.copy()
                refPoints1 = corners
                
                # Buscamos la segunda imagen unos frames más adelante (j = i + 10)
                j = i + 10
                if j <= total_frames - 5:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                    ret2, frame2 = cap.read()
                    
                    if ret2:
                        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                        ret_corners2, corners2 = cv2.findChessboardCorners(gray2, self.chessboard_size, None)
                        
                        if ret_corners2:
                            imagen2 = frame2.copy()
                            refPoints2 = corners2
                            break # Encontramos ambas imágenes válidas, salimos del bucle
                            
            # Volvemos a la posición del salto si no hemos terminado
            i += salto
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            
        cap.release()
        
        if imagen1 is None or imagen2 is None:
            raise ValueError("No se encontraron dos fotogramas válidos con el patrón de calibración visible.")
            
        return imagen1, imagen2, refPoints1, refPoints2

# --- Ejemplo de uso ---
# motor = MotorAnalisisHeridas(chessboard_size=(9, 6))
# img1, img2, pts1, pts2 = motor.escoger_2_imagenes_desde_video("ruta_al_video.mp4")