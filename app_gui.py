import flet as ft
import cv2
import os
import numpy as np
import base64

# Importamos tu motor matemático
from config_camara import ConfigCamara
from adquisicion import MotorAnalisisHeridas
from motor_analisis import AnalizadorTejidos
from reconstruccion_3d import Reconstructor3D

class NurseCareApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "NurseCare - Análisis Avanzado"
        self.page.theme_mode = "light" 
        self.page.window.width = 1300
        self.page.window.height = 850
        self.page.padding = 20
        self.page.bgcolor = "#f4f6f8"
        
        self.config = ConfigCamara()
        self.adquisicion = MotorAnalisisHeridas(chessboard_size=(9, 6))
        self.analizador_2d = AnalizadorTejidos(n_colores=3)
        self.motor_3d = Reconstructor3D(self.config)
        
        self.img1_bgr = None
        self.mascara_roi = None
        
        self.init_ui()

    def convertir_cv2_a_base64(self, imagen_cv2):
        """Convierte una matriz de OpenCV a texto Base64 para inyectar en la web"""
        _, buffer = cv2.imencode('.jpg', imagen_cv2)
        imagen_texto = base64.b64encode(buffer).decode('utf-8')
        return imagen_texto

    def init_ui(self):
        # 1. Contenedor de la Imagen
        self.view_imagen = ft.Image(
            src="https://via.placeholder.com/800x600.png?text=Esperando+Video...", 
            fit="contain",
            expand=True
        )
        
        self.contenedor_visor = ft.Container(
            content=self.view_imagen,
            alignment=ft.Alignment(0, 0),
            expand=True,
            bgcolor="white",
            padding=10
        )

        # 2. Etiquetas de Resultados
        self.txt_estado = ft.Text("Estado: Esperando acción...", italic=True, color="blue")
        self.lbl_area = ft.Text("Área Total: -- mm²", size=18, weight="bold")
        self.lbl_nec = ft.Text("Tejido Necrótico: --%", color="black")
        self.lbl_esf = ft.Text("Tejido Esfacelar: --%", color="orange")
        self.lbl_gra = ft.Text("Tejido Granulación: --%", color="red")
        self.lbl_puntos3d = ft.Text("Puntos 3D: --", color="blue")

        # 3. Botones ULTRA simples
        self.btn_cargar = ft.ElevatedButton(
            content=ft.Text("1. Cargar Vídeo", weight="bold"),
            on_click=self.cargar_video_click
        )

        self.btn_analizar = ft.ElevatedButton(
            content=ft.Text("2. Ejecutar Análisis", weight="bold"),
            on_click=self.procesar_analisis,
            disabled=True
        )

        # 4. Panel de Control Lateral
        self.panel_reporte = ft.Container(
            content=ft.Column([
                ft.Text("Panel de Control", size=22, weight="bold"),
                self.btn_cargar,
                ft.Divider(height=30),
                
                ft.Text("Reporte Clínico", size=20, weight="bold"),
                self.txt_estado,
                self.lbl_area,
                self.lbl_nec,
                self.lbl_esf,
                self.lbl_gra,
                self.lbl_puntos3d,
                ft.Divider(height=30),
                self.btn_analizar
            ], spacing=10),
            width=350,
            padding=20,
            bgcolor="white"
        )

        # 5. Vista principal
        vista_principal = ft.Row([
            self.contenedor_visor,
            self.panel_reporte
        ], expand=True, spacing=20)

        self.page.add(
            ft.AppBar(
                title=ft.Text("NurseCare - Análisis Geométrico"),
                bgcolor="blue900",
                color="white"
            ),
            vista_principal
        )
        self.page.update()

    def cargar_video_click(self, e):
        ruta_test = "tu_video_de_prueba.mp4"
        
        if not os.path.exists(ruta_test):
            self.txt_estado.value = f"Error: No se encuentra '{ruta_test}'"
            self.txt_estado.color = "red"
            self.page.update()
            return

        self.txt_estado.value = "Procesando vídeo..."
        self.txt_estado.color = "blue"
        self.page.update()

        try:
            img1, img2, self.pts1, self.pts2 = self.adquisicion.escoger_2_imagenes_desde_video(ruta_test)
            self.img1_bgr = img1
            self.img2_bgr = img2
            
            # --- CORRECCIÓN: Simplemente inyectamos el Base64. Flet le dará prioridad. ---
            # self.view_imagen.src = None <-- ESTA LÍNEA PROVOCABA EL ERROR, LA HEMOS QUITADO
            self.view_imagen.src_base64 = self.convertir_cv2_a_base64(img1)
            
            self.txt_estado.value = "Vídeo cargado. Listo para analizar."
            self.txt_estado.color = "green"
            self.btn_analizar.disabled = False 
            
        except Exception as ex:
            self.txt_estado.value = f"Error: {str(ex)}"
            self.txt_estado.color = "red"
            
        self.page.update()

    def procesar_analisis(self, e):
        self.txt_estado.value = "Calculando IA K-Means y Nube 3D..."
        self.txt_estado.color = "blue"
        self.btn_analizar.disabled = True 
        self.page.update()
        
        alto, ancho = self.img1_bgr.shape[:2]
        cx, cy = ancho // 2, alto // 2
        r = 150
        puntos_contorno = [(cx-r, cy-r), (cx+r, cy-r), (cx+r, cy+r), (cx-r, cy+r)]

        img_recortada, mascara_roi = self.analizador_2d.aplicar_mascara_poligono(self.img1_bgr, puntos_contorno)
        _, segmentos = self.analizador_2d.segmentar_kmeans(img_recortada)
        tej_nec, tej_gra, tej_esf = self.analizador_2d.ordenar_tejidos(segmentos)
        
        resultados = self.analizador_2d.calcular_areas_y_porcentajes(mascara_roi, tej_nec, tej_gra, tej_esf, factor_escala=0.25)
        
        mascara_8u = (mascara_roi * 255).astype(np.uint8)
        nube_puntos, _ = self.motor_3d.calcular_3d(self.img1_bgr, self.img2_bgr, self.pts1, self.pts2, mascara_8u)

        # Sumamos los tres canales de tejido segmentado para crear una imagen a color
        mapa_tejidos = cv2.addWeighted(tej_nec, 1, tej_gra, 1, 0)
        mapa_tejidos = cv2.addWeighted(mapa_tejidos, 1, tej_esf, 1, 0)
        
        # Actualizar imagen mostrada
        self.view_imagen.src_base64 = self.convertir_cv2_a_base64(mapa_tejidos)

        # Actualizar textos
        self.lbl_area.value = f"Área Total: {resultados['areas']['total']:.2f} mm²"
        self.lbl_nec.value = f"Tejido Necrótico: {resultados['porcentajes']['necrotico']:.1f}%"
        self.lbl_esf.value = f"Tejido Esfacelar: {resultados['porcentajes']['esfacelo']:.1f}%"
        self.lbl_gra.value = f"Tejido Granulación: {resultados['porcentajes']['granulacion']:.1f}%"
        self.lbl_puntos3d.value = f"Puntos 3D: {len(nube_puntos)} extraídos (zona prueba)"
        
        self.txt_estado.value = "Análisis completado."
        self.txt_estado.color = "green"
        self.btn_analizar.disabled = False
        self.page.update()

def main(page: ft.Page):
    NurseCareApp(page)

if __name__ == "__main__":
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8552)