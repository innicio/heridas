import cv2
import numpy as np
import time

# Importamos nuestros módulos modulares
from config_camara import ConfigCamara
from adquisicion import MotorAnalisisHeridas
from motor_analisis import AnalizadorTejidos
from reconstruccion_3d import Reconstructor3D

def seleccionar_poligono_simulado(imagen):
    """
    Función puente temporal para entornos sin GUI (headless).
    Simula la delimitación manual de la herida aportando un polígono cuadrado automatizado
    en el centro de la imagen.
    """
    alto, ancho = imagen.shape[:2]
    cx, cy = ancho // 2, alto // 2
    r = 150  # Radio del cuadro de simulación
    
    # Generamos un polígono cerrado simulado en el centro
    puntos = [
        (cx - r, cy - r),
        (cx + r, cy - r),
        (cx + r, cy + r),
        (cx - r, cy + r)
    ]
    print(f"[INFO GUI] Entorno Headless detectado. Simulando polígono de interés en el centro: {puntos}")
    return puntos

def main():
    print("--- INICIANDO MOTOR DE ANÁLISIS CLÍNICO ---")
    ruta_video = "tu_video_de_prueba.mp4" # <--- ASEGÚRATE DE QUE TIENE ESTE NOMBRE
    
    # 1. Inicializar configuración e instancias
    config = ConfigCamara()
    adquisicion = MotorAnalisisHeridas(chessboard_size=(9, 6))
    analizador_2d = AnalizadorTejidos(n_colores=3)
    motor_3d = Reconstructor3D(config, tamano_cuadro_mm=5.0)

    # 2. Extraer fotogramas válidos
    print("Buscando patrón de calibración en el vídeo...")
    t_inicio = time.time()
    img1, img2, pts1, pts2 = adquisicion.escoger_2_imagenes_desde_video(ruta_video)
    print(f"Fotogramas extraídos en {time.time() - t_inicio:.2f} segundos.")

    # 3. Interacción simulada (Aislamos el problema de la ventana gráfica)
    puntos_contorno = seleccionar_poligono_simulado(img1)
    
    # 4. Procesamiento 2D (Segmentación de tejidos)
    print("Analizando composición de tejidos mediante K-Means...")
    img_recortada, mascara_roi = analizador_2d.aplicar_mascara_poligono(img1, puntos_contorno)
    etiquetas, segmentos = analizador_2d.segmentar_kmeans(img_recortada)
    tej_nec, tej_gra, tej_esf = analizador_2d.ordenar_tejidos(segmentos)
    
    # Factor de escala (mm/píxel) simulado para la prueba analítica
    factor_escala = 0.25 
    resultados = analizador_2d.calcular_areas_y_porcentajes(
        mascara_roi, tej_nec, tej_gra, tej_esf, factor_escala
    )

    # 5. Procesamiento 3D (Extracción de la nube de puntos)
    print("Calculando volumetría y triangulación 3D (SIFT)...")
    mascara_8u = (mascara_roi * 255).astype(np.uint8)
    
    t_3d_inicio = time.time()
    nube_puntos, colores = motor_3d.calcular_3d(img1, img2, pts1, pts2, mascara_8u)
    duracion_3d = time.time() - t_3d_inicio

    # --- MOSTRAR REPORTE ANALÍTICO POR CONSOLA ---
    print("\n" + "="*40)
    print("📊 REPORTE CLÍNICO GENERADO EN PYTHON")
    print("="*40)
    if resultados:
        print(f"Área Total de la lesión: {resultados['areas']['total']:.2f} mm²")
        print(f"-> Tejido Necrótico:     {resultados['porcentajes']['necrotico']:.1f}%")
        print(f"-> Tejido de Esfacelo:   {resultados['porcentajes']['esfacelo']:.1f}%")
        print(f"-> Tejido Granulación:   {resultados['porcentajes']['granulacion']:.1f}%")
    else:
        print("Área de la lesión: No se han podido computar las áreas.")
    print("-" * 40)
    print(f"Reconstrucción 3D:       {len(nube_puntos)} puntos espaciales calculados.")
    print(f"Tiempo de cálculo 3D:    {duracion_3d:.2f} segundos.")
    print("="*40)
    
    # Nota: Guardamos los resultados numéricos de la nube en un archivo para no perderlos
    if len(nube_puntos) > 0:
        np.savez('resultado_3d_test.npz', puntos=nube_puntos, colores=colores)
        print("[SISTEMA] Nube de puntos exportada correctamente a 'resultado_3d_test.npz'")

if __name__ == "__main__":
    main()