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

    def escoger_2_imagenes_desde_video(self, ruta_video):
        """
        Traducción directa del algoritmo de MATLAB f_escoger_2_imagenes_desde_video.
        Garantiza un salto de fotogramas (baseline) para el paralaje topográfico.
        """
        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise ValueError(f"Error: No se pudo abrir el vídeo '{ruta_video}'")

        # Emulando info.Duration * info.FrameRate
        frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Variables de control idénticas al script original
        i = 70       # Fotograma de inicio (REVISAR según el vídeo, como anotaste)
        salto = 2    # Avance en la búsqueda
        gap_estereo = 10 # j = i + 10 (La clave del relieve 3D)

        imagen1 = None
        imagen2 = None
        refPoints1 = None
        refPoints2 = None

        while i <= frames_total - salto - gap_estereo:
            # Nos posicionamos en el frame 'i' y leemos
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret1, frame1 = cap.read()
            if not ret1:
                break

            # Búsqueda del patrón (equivalente a detectCheckerboardPoints)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            encontrado1, esquinas1 = cv2.findChessboardCorners(gray1, self.chessboard_size, None)

            # Si detecta las esquinas, comprobamos que sean exactamente 54
            if encontrado1 and esquinas1 is not None and len(esquinas1) == 54:
                
                # Si cumple, miramos 10 frames hacia el futuro (j)
                j = i + gap_estereo
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                ret2, frame2 = cap.read()

                if ret2:
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                    encontrado2, esquinas2 = cv2.findChessboardCorners(gray2, self.chessboard_size, None)

                    # Verificamos los 54 puntos en la segunda imagen
                    if encontrado2 and esquinas2 is not None and len(esquinas2) == 54:
                        
                        # Guardamos resultados y simulamos el k=k+1 y cierre de bucle
                        imagen1 = frame1
                        imagen2 = frame2
                        # Refinamos las esquinas para máxima precisión subpíxel (opcional pero recomendado)
                        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        refPoints1 = cv2.cornerSubPix(gray1, esquinas1, (11, 11), (-1, -1), criterios)
                        refPoints2 = cv2.cornerSubPix(gray2, esquinas2, (11, 11), (-1, -1), criterios)
                        
                        break # Termina el while (equivalente a i = frames - salto en MATLAB)

            # Avanzamos el contador
            i += salto

        cap.release()

        if imagen1 is None or imagen2 is None:
            raise RuntimeError("El algoritmo de salto no encontró 2 fotogramas válidos separados por 10 frames.")

        return imagen1, imagen2, refPoints1, refPoints2

# --- Ejemplo de uso ---
# motor = MotorAnalisisHeridas(chessboard_size=(9, 6))
# img1, img2, pts1, pts2 = motor.escoger_2_imagenes_desde_video("ruta_al_video.mp4")