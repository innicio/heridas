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
        self.page.title = "NurseCare - Validación Clínica 2D"
        self.page.theme_mode = "light" 
        self.page.window.width = 1400 
        self.page.window.height = 900
        self.page.padding = 10
        self.page.bgcolor = "#f0f2f5"
        
        # Inicializar motor matemático
        self.config = ConfigCamara()
        self.adquisicion = MotorAnalisisHeridas(chessboard_size=(9, 6))
        self.analizador_2d = AnalizadorTejidos(n_colores=3)
        self.motor_3d = Reconstructor3D(self.config)
        
        self.img1_bgr = None
        
        self.init_ui()

    def convertir_cv2_a_base64(self, imagen_cv2):
        """Convierte OpenCV BGR a Base64 JPG para la web"""
        try:
            _, buffer = cv2.imencode('.jpg', imagen_cv2)
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error convirtiendo imagen: {e}")
            return ""

    def crear_visor_imagen(self, titulo):
        """Crea un contenedor estándar para mostrar imágenes (Libre de src_base64)"""
        img_control = ft.Image(
            src="", 
            fit="contain",
            expand=True,
            visible=False 
        )
        txt_placeholder = ft.Text(f"Esperando {titulo}...", color="grey500", size=16)
        
        contenedor = ft.Container(
            content=ft.Column([
                ft.Text(titulo, weight="bold", size=16, color="bluegrey800"),
                ft.Container(
                    content=ft.Stack([txt_placeholder, img_control]),
                    alignment=ft.Alignment(0, 0),
                    expand=True,
                    bgcolor="grey100",
                    border_radius=8,
                    padding=10
                )
            ]),
            expand=True,
            bgcolor="white",
            padding=15,
            border_radius=10,
            shadow=ft.BoxShadow(spread_radius=1, blur_radius=3, color="black12")
        )
        contenedor.img_control = img_control
        contenedor.txt_placeholder = txt_placeholder
        return contenedor

    def init_ui(self):
        # --- ZONA VISUAL (Side-by-Side) ---
        self.visor_original = self.crear_visor_imagen("Foto Original (BGR)")
        self.visor_segmentacion = self.crear_visor_imagen("Validación IA (K-Means)")
        
        self.zona_imagenes = ft.Row([
            self.visor_original,
            self.visor_segmentacion
        ], expand=True, spacing=15)

        # --- PANEL DE CONTROL ---
        self.txt_estado = ft.Text("Estado: Esperando acción...", italic=True, color="blue")
        self.lbl_area = ft.Text("Área Total: -- mm²", size=18, weight="bold")
        self.lbl_nec = ft.Text("Necrosis (Negro): --%", color="black")
        self.lbl_esf = ft.Text("Esfacelo (Amarillo): --%", color="#cca300")
        self.lbl_gra = ft.Text("Granulación (Rojo): --%", color="red")
        self.lbl_puntos3d = ft.Text("Puntos 3D SIFT: --", color="blue")

        self.btn_cargar = ft.ElevatedButton(
            content=ft.Text("1. Cargar Vídeo Prueba", weight="bold"),
            on_click=self.cargar_video_click,
            style=ft.ButtonStyle(bgcolor="blue800", color="white")
        )

        self.btn_analizar = ft.ElevatedButton(
            content=ft.Text("2. Validar Segmentación 2D", weight="bold"),
            on_click=self.procesar_analisis,
            disabled=True,
            style=ft.ButtonStyle(bgcolor="green700", color="white")
        )

        self.panel_control = ft.Container(
            content=ft.Column([
                ft.Text("Panel Clínico", size=22, weight="bold"),
                self.btn_cargar,
                ft.Divider(height=20),
                ft.Text("Resultados", size=18, weight="w600"),
                self.txt_estado,
                ft.Card(
                    # FIX: Eliminado 'color' de Card, movido a 'bgcolor' del Container interno
                    content=ft.Container(
                        bgcolor="grey50", 
                        padding=15,
                        content=ft.Column([
                            self.lbl_area,
                            self.lbl_nec,
                            self.lbl_esf,
                            self.lbl_gra,
                            ft.Divider(),
                            self.lbl_puntos3d
                        ], spacing=8)
                    )
                ),
                self.btn_analizar
            ], spacing=10),
            width=320,
            padding=20,
            bgcolor="white",
            border_radius=10,
            shadow=ft.BoxShadow(spread_radius=1, blur_radius=3, color="black12")
        )

        # --- DISTRIBUCIÓN FINAL ---
        self.page.add(
            ft.AppBar(
                title=ft.Text("NurseCare - Herramienta de Validación Clínica"),
                bgcolor="bluegrey800",
                color="white"
            ),
            ft.Row([
                self.zona_imagenes,
                self.panel_control
            ], expand=True, spacing=15)
        )
        self.page.update()

    def actualizar_imagen(self, visor, imagen_cv2):
        """Inyecta de forma segura una imagen OpenCV usando el estándar Data URI"""
        b64 = self.convertir_cv2_a_base64(imagen_cv2)
        if b64:
            visor.img_control.src = f"data:image/jpeg;base64,{b64}"
            visor.img_control.visible = True
            visor.txt_placeholder.visible = False
            self.page.update()

    def cargar_video_click(self, e):
        print("[SISTEMA] Clic en Cargar")
        ruta_test = "tu_video_de_prueba.mp4"
        
        if not os.path.exists(ruta_test):
            self.txt_estado.value = f"Error: No está '{ruta_test}'"
            self.txt_estado.color = "red"
            self.page.update()
            return

        self.txt_estado.value = "Buscando tablero..."
        self.txt_estado.color = "blue"
        self.btn_cargar.disabled = True
        self.page.update()

        try:
            img1, img2, self.pts1, self.pts2 = self.adquisicion.escoger_2_imagenes_desde_video(ruta_test)
            self.img1_bgr = img1
            self.img2_bgr = img2
            
            self.actualizar_imagen(self.visor_original, img1)
            
            self.visor_segmentacion.img_control.visible = False
            self.visor_segmentacion.txt_placeholder.visible = True

            self.txt_estado.value = "Foto original cargada. Listo para segmentar."
            self.txt_estado.color = "green"
            self.btn_analizar.disabled = False 
            
        except Exception as ex:
            self.txt_estado.value = f"Error adquisición: {str(ex)}"
            self.txt_estado.color = "red"
            print(f"[ERROR] {ex}")
            
        self.btn_cargar.disabled = False
        self.page.update()

    def procesar_analisis(self, e):
        print("[SISTEMA] Clic en Analizar 2D")
        self.txt_estado.value = "Ejecutando K-Means..."
        self.txt_estado.color = "blue"
        self.btn_analizar.disabled = True 
        self.page.update()
        
        # 1. SOLUCIÓN AL RECORTE: Hacemos que el polígono sea (casi) toda la foto
        alto, ancho = self.img1_bgr.shape[:2]
        margen = 20 # Dejamos solo 20 píxeles de margen en los bordes
        puntos_contorno = [
            (margen, margen), 
            (ancho - margen, margen), 
            (ancho - margen, alto - margen), 
            (margen, alto - margen)
        ]

        img_recortada, mascara_roi = self.analizador_2d.aplicar_mascara_poligono(self.img1_bgr, puntos_contorno)
        _, segmentos = self.analizador_2d.segmentar_kmeans(img_recortada)
        tej_nec, tej_gra, tej_esf = self.analizador_2d.ordenar_tejidos(segmentos)
        
        mascara_8u = (mascara_roi * 255).astype(np.uint8)
        nube_puntos, _ = self.motor_3d.calcular_3d(self.img1_bgr, self.img2_bgr, self.pts1, self.pts2, mascara_8u)

        # 2. SOLUCIÓN A LA SEGMENTACIÓN: Crear mapa de "Colores Falsos"
        mapa_colores = np.zeros_like(img_recortada)
        
        # Detectamos dónde hay píxeles en cada capa de tejido
        mask_nec = cv2.cvtColor(tej_nec, cv2.COLOR_BGR2GRAY) > 0
        mask_gra = cv2.cvtColor(tej_gra, cv2.COLOR_BGR2GRAY) > 0
        mask_esf = cv2.cvtColor(tej_esf, cv2.COLOR_BGR2GRAY) > 0

        # Pintamos la imagen resultante con colores puros llamativos (OpenCV usa BGR)
        mapa_colores[mask_nec] = [50, 50, 50]    # Gris oscuro (Necrosis)
        mapa_colores[mask_esf] = [0, 255, 255]   # Amarillo puro (Esfacelo)
        mapa_colores[mask_gra] = [0, 0, 255]     # Rojo puro (Granulación)
        
        self.actualizar_imagen(self.visor_segmentacion, mapa_colores)

        # Actualizar textos del reporte
        resultados = self.analizador_2d.calcular_areas_y_porcentajes(mascara_roi, tej_nec, tej_gra, tej_esf, factor_escala=0.25)

        self.lbl_area.value = f"Área total zona interés: {resultados['areas']['total']:.2f} mm²"
        self.lbl_nec.value = f"Necrosis (Gris): {resultados['porcentajes']['necrotico']:.1f}%"
        self.lbl_esf.value = f"Esfacelo (Amarillo): {resultados['porcentajes']['esfacelo']:.1f}%"
        self.lbl_gra.value = f"Granulación (Rojo): {resultados['porcentajes']['granulacion']:.1f}%"
        self.lbl_puntos3d.value = f"Puntos 3D SIFT: {len(nube_puntos)} extraídos"
        
        self.txt_estado.value = "Análisis 2D completado."
        self.txt_estado.color = "green"
        self.btn_analizar.disabled = False
        self.page.update()

def main(page: ft.Page):
    NurseCareApp(page)

if __name__ == "__main__":
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8552)