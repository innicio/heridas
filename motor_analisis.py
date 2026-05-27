import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

class AnalizadorTejidos:
    def __init__(self, n_colores=3):
        """
        Inicializa el motor de segmentación y análisis de tejidos.
        :param n_colores: Número de clústeres para K-means (por defecto 3: Necrosis, Esfacelo, Granulación).
        """
        self.n_colores = n_colores

    def segmentar_kmeans(self, imagen_recortada):
        """
        Aplica el algoritmo K-Means en el espacio de color L*a*b* Equivalente a f_segmentar_kmean.m de MATLAB.
        """
        # 1. Convertir la imagen de BGR (OpenCV) al espacio de color CIE L*a*b*
        imagen_lab = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2LAB)
        
        # 2. Extraer los canales a* y b* (índices 1 y 2 en Python)
        # El canal 0 es la Luminosidad (L*), que se descarta para evitar interferencias por iluminación
        ab_canales = imagen_lab[:, :, 1:3]
        
        alto, ancho, canales = ab_canales.shape
        datos_pixeles = ab_canales.reshape(alto * ancho, canales).astype(np.float32)
        
        # 3. Ejecutar el agrupamiento K-Means
        # n_init=3 emula el parámetro 'Replicates', 3 de MATLAB para evitar mínimos locales
        kmeans = KMeans(n_clusters=self.n_colores, n_init=3, random_state=42)
        etiquetas_clúster = kmeans.fit_predict(datos_pixeles)
        
        # Reestructurar las etiquetas a las dimensiones originales de la imagen
        pixel_labels = etiquetas_clúster.reshape(alto, ancho)
        
        # 4. Separar la imagen original en 3 máscaras de color individuales (seg1, seg2, seg3)
        imagenes_segmentadas = []
        for k in range(self.n_colores):
            mascara_color = imagen_recortada.copy()
            # En Python las etiquetas van de 0 a (n-1)
            mascara_color[pixel_labels != k] = 0
            imagenes_segmentadas.append(mascara_color)
            
        return pixel_labels, imagenes_segmentadas

    def ordenar_tejidos(self, imagenes_segmentadas):
        """
        Clasifica automáticamente los segmentos en Necrótico, Granulación y Esfacelo.
        Equivalente mejorado de f_ordenar_kmeans_corto.m utilizando el umbral de Otsu.
        """
        seg0, seg1, seg2 = imagenes_segmentadas
        
        # Calcular el umbral de Otsu para cada segmento (equivalente a multithresh de MATLAB)
        # Convertimos a escala de grises para obtener la intensidad del umbral
        t0 = threshold_otsu(cv2.cvtColor(seg0, cv2.COLOR_BGR2GRAY)) if np.any(seg0) else 0
        t1 = threshold_otsu(cv2.cvtColor(seg1, cv2.COLOR_BGR2GRAY)) if np.any(seg1) else 0
        t2 = threshold_otsu(cv2.cvtColor(seg2, cv2.COLOR_BGR2GRAY)) if np.any(seg2) else 0
        
        # Estructurar los datos para ordenarlos por su nivel de umbral
        estructuras = [
            {'segmentacion': seg0, 'umbral': t0},
            {'segmentacion': seg1, 'umbral': t1},
            {'segmentacion': seg2, 'umbral': t2}
        ]
        
        # Ordenar de menor a mayor umbral
        estructuras_ordenadas = sorted(estructuras, key=lambda x: x['umbral'])
        
        # Clasificación clínica basada en el orden analítico de tus umbrales originales:
        # El tejido con menor umbral (más oscuro) se asigna a necrosis
        tejido_necrotico = estructuras_ordenadas[0]['segmentacion']
        tejido_granulacion = estructuras_ordenadas[1]['segmentacion']
        tejido_esfacelo = estructuras_ordenadas[2]['segmentacion']
        
        return tejido_necrotico, tejido_granulacion, tejido_esfacelo

    def aplicar_mascara_poligono(self, imagen, puntos_poligono):
        """
        Equivalente a f_recortar_imagen.m.
        Aísla la región de la herida delimitada por el usuario.
        
        :param imagen: Imagen original leída con cv2.imread o desde vídeo.
        :param puntos_poligono: Lista de tuplas con las coordenadas [(x1,y1), (x2,y2)...]
        :return: imagen_recortada, mascara_booleana
        """
        alto, ancho = imagen.shape[:2]
        
        # 1. Crear una máscara negra (ceros) del tamaño de la imagen
        mascara = np.zeros((alto, ancho), dtype=np.uint8)
        
        # Formatear los puntos para OpenCV: matriz de enteros de forma [N, 1, 2]
        pts = np.array(puntos_poligono, np.int32).reshape((-1, 1, 2))
        
        # 2. Rellenar el polígono de blanco (255) en la máscara
        cv2.fillPoly(mascara, [pts], 255)
        
        # 3. Crear la imagen de fondo gris (0.94 * 255 = 240 en tu MATLAB original)
        fondo_gris = np.full(imagen.shape, 240, dtype=np.uint8)
        
        # 4. Combinar: Dentro del polígono guardamos la imagen, fuera ponemos el fondo gris
        # Expandimos la máscara a 3 dimensiones para que coincida con los canales RGB
        mascara_3d = mascara[:, :, np.newaxis] == 255
        imagen_recortada = np.where(mascara_3d, imagen, fondo_gris)
        
        # Devolvemos la imagen procesada y la máscara booleana (True dentro de la herida)
        mascara_booleana = mascara == 255
        return imagen_recortada, mascara_booleana

    def calcular_areas_y_porcentajes(self, mascara_roi, mascara_necrotico, mascara_granulacion, mascara_esfacelo, factor_escala):
        """
        Equivalente a f_calcular_areas.m
        Calcula el área real (ej. en cm²) basándose en la calibración y el porcentaje de cada tejido.
        """
        # Función auxiliar para contar píxeles de color dentro de la región de interés (ROI)
        def contar_pixeles_tejido(segmento):
            # Convertimos a gris: si el valor es > 0, es que hay tejido asignado a ese píxel
            gris = cv2.cvtColor(segmento, cv2.COLOR_BGR2GRAY)
            # Contamos cuántos píxeles cumplen ambas condiciones: tienen color Y están dentro del polígono
            return np.sum((gris > 0) & mascara_roi)

        # 1. Contar píxeles
        pix_necrotico = contar_pixeles_tejido(mascara_necrotico)
        pix_granulacion = contar_pixeles_tejido(mascara_granulacion)
        pix_esfacelo = contar_pixeles_tejido(mascara_esfacelo)
        
        # 2. Multiplicar por el factor de escala al cuadrado (pasar de píxeles² a cm²)
        # Si factor_escala es mm/píxel, el resultado estará en mm²
        factor_area = factor_escala ** 2
        area_necrotico = pix_necrotico * factor_area
        area_granulacion = pix_granulacion * factor_area
        area_esfacelo = pix_esfacelo * factor_area
        
        area_total = area_necrotico + area_granulacion + area_esfacelo
        
        # 3. Calcular porcentajes (protección contra división por cero)
        if area_total == 0:
            return None # Evitar errores si se selecciona un área vacía
            
        porc_necrotico = (area_necrotico / area_total) * 100
        porc_granulacion = (area_granulacion / area_total) * 100
        porc_esfacelo = (area_esfacelo / area_total) * 100
        
        # Devolvemos los datos organizados en un diccionario para poder leerlos fácil desde la UI
        return {
            'areas': {
                'necrotico': area_necrotico,
                'granulacion': area_granulacion,
                'esfacelo': area_esfacelo,
                'total': area_total
            },
            'porcentajes': {
                'necrotico': porc_necrotico,
                'granulacion': porc_granulacion,
                'esfacelo': porc_esfacelo
            }
        }