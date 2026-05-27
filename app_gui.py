import flet as ft
import flet_webview as fvw
import cv2
import os
import numpy as np
import base64
import warnings
import tempfile # Añadido para crear el archivo HTML temporal

# Silenciamos de forma global los avisos de depreciación de Flet 0.80+
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importamos Plotly y los motores de interpolación geométrica
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

# Importamos tu motor matemático
from config_camara import ConfigCamara
from adquisicion import MotorAnalisisHeridas
from motor_analisis import AnalizadorTejidos
from reconstruccion_3d import Reconstructor3D

class NurseCareApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "NurseCare - Análisis Clínico de Heridas"
        self.page.theme_mode = "light" 
        self.page.padding = 20
        self.page.bgcolor = "#f0f2f5"
        self.page.scroll = "auto"
        
        # Motores de análisis
        self.config = ConfigCamara()
        self.adquisicion = MotorAnalisisHeridas(chessboard_size=(9, 6))
        self.analizador_2d = AnalizadorTejidos(n_colores=3)
        self.motor_3d = Reconstructor3D(self.config)
        
        # Variables de estado
        self.img1_bgr = None
        self.img1_original = None  
        self.img2_bgr = None
        self.pts1 = None
        self.pts2 = None
        self.puntos_reales = []    
        self.html_file_path = None # Guardar la ruta del archivo HTML temporal
        
        self.ANCHO_VISOR_GRANDE = 800
        self.ALTO_VISOR_GRANDE = 600

        self.init_ui()

    def convertir_cv2_a_base64(self, imagen_cv2):
        try:
            _, buffer = cv2.imencode('.jpg', imagen_cv2)
            b64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
        except Exception:
            return ""

    def init_ui(self):
        # ==========================================
        # 1. INICIALIZAR TODOS LOS VISORES PRIMERO
        # ==========================================
        self.img_recorte = ft.Image(
            src="", 
            width=self.ANCHO_VISOR_GRANDE,
            height=self.ALTO_VISOR_GRANDE,
            fit="contain",
            visible=False 
        )
        self.txt_placeholder_recorte = ft.Text("Carga un vídeo para comenzar...", color="grey500", size=18)
        
        self.contenedor_pizarra = ft.Container(
            content=ft.Stack([self.txt_placeholder_recorte, self.img_recorte]),
            alignment=ft.Alignment(0, 0),
            width=self.ANCHO_VISOR_GRANDE,
            height=self.ALTO_VISOR_GRANDE,
            bgcolor="white",
            border_radius=10,
        )

        self.detector_trazado = ft.GestureDetector(
            content=self.contenedor_pizarra,
            on_pan_update=self.trazar_borde,
            on_pan_end=self.soltar_raton, 
            on_tap_down=self.clic_suelto_borde
        )

        # Visores del Dashboard de Resultados
        self.visor_res_recorte = ft.Image(src="", expand=True, fit="contain")
        self.visor_res_segmentado = ft.Image(src="", expand=True, fit="contain")
        
        # Inicializamos el visor como un simple texto al arrancar la app
        self.visor_res_3d = ft.Container(
            content=ft.Text("Malla 3D en construcción...", color="grey500"),
            alignment=ft.Alignment(0, 0),
            expand=True
        )

        # ==========================================
        # 2. DEFINIR BOTONES Y ESTRUCTURA DE TARJETAS
        # ==========================================
        self.btn_cargar = ft.ElevatedButton(
            content=ft.Text("1. Cargar Vídeo", weight="bold"),
            on_click=self.cargar_video_click,
            style=ft.ButtonStyle(bgcolor="blue800", color="white")
        )
        
        self.btn_limpiar = ft.ElevatedButton(
            content=ft.Text("Borrar Trazado", weight="bold"),
            on_click=self.limpiar_recorte_click,
            disabled=True
        )

        self.btn_analizar = ft.ElevatedButton(
            content=ft.Text("2. Confirmar Recorte y Analizar", weight="bold"),
            on_click=self.procesar_analisis,
            disabled=True,
            style=ft.ButtonStyle(bgcolor="green700", color="white")
        )

        def crear_tarjeta(titulo, contenido):
            return ft.Container(
                content=ft.Column([
                    ft.Text(titulo, weight="bold", size=16, color="bluegrey800"),
                    ft.Container(content=contenido, expand=True, alignment=ft.Alignment(0, 0))
                ]),
                bgcolor="white", padding=15, border_radius=10, expand=True, height=400,
                shadow=ft.BoxShadow(spread_radius=1, blur_radius=3, color="black12")
            )

        tarjetas_resultados = ft.Row([
            crear_tarjeta("1. Recorte Real 2D", self.visor_res_recorte),
            crear_tarjeta("2. Segmentación Aislada", self.visor_res_segmentado),
            crear_tarjeta("3. Malla 3D Texturizada", self.visor_res_3d)
        ], expand=True, spacing=20)

        # Etiquetas para las métricas
        self.lbl_area = ft.Text("-- mm²", size=20, weight="bold", color="blue900")
        self.lbl_nec = ft.Text("--%", weight="bold", color="black")
        self.lbl_esf = ft.Text("--%", weight="bold", color="#cca300")
        self.lbl_gra = ft.Text("--%", weight="bold", color="red")
        
        panel_metricas = ft.Container(
            content=ft.Row([
                ft.Column([ft.Text("Área Total"), self.lbl_area]),
                ft.Column([ft.Text("Necrosis (Gris)"), self.lbl_nec]),
                ft.Column([ft.Text("Esfacelo (Amarillo)"), self.lbl_esf]),
                ft.Column([ft.Text("Granulación (Rojo)"), self.lbl_gra])
            ], alignment=ft.MainAxisAlignment.SPACE_EVENLY),
            bgcolor="white", padding=20, border_radius=10,
            shadow=ft.BoxShadow(spread_radius=1, blur_radius=3, color="black12")
        )

        self.btn_volver = ft.ElevatedButton(
            content=ft.Text("Volver al Recorte", weight="bold"),
            on_click=self.volver_al_recorte,
            style=ft.ButtonStyle(bgcolor="bluegrey700", color="white")
        )

        # ==========================================
        # 3. CONSTRUIR PANTALLAS Y AÑADIR A LA PÁGININA
        # ==========================================
        self.pantalla_recorte = ft.Column([
            ft.Text("Paso 1: Delimitación de la Úlcera", size=24, weight="bold", color="bluegrey900"),
            ft.Text("Haz clic y ARRASTRA el ratón bordeando la herida para generar el contorno.", size=16),
            ft.Row([self.btn_cargar, self.btn_limpiar, self.btn_analizar]),
            ft.Container(height=20),
            ft.Container(
                content=self.detector_trazado,
                shadow=ft.BoxShadow(spread_radius=1, blur_radius=5, color="black12")
            )
        ], alignment=ft.MainAxisAlignment.START, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        self.pantalla_resultados = ft.Column([
            ft.Row([
                ft.Text("Paso 2: Resultados del Análisis", size=24, weight="bold", color="bluegrey900"),
                self.btn_volver
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            tarjetas_resultados,
            ft.Container(height=20), 
            panel_metricas
        ], visible=False)

        self.page.add(self.pantalla_recorte, self.pantalla_resultados)
        self.page.update()

    # ==========================================
    # LÓGICA DE EXTRACCIÓN Y TRAZADO
    # ==========================================
    def extraer_coordenadas(self, e):
        if hasattr(e, 'local_position') and e.local_position is not None:
            return e.local_position.x, e.local_position.y
        elif hasattr(e, 'local_x') and e.local_x is not None:
            return float(e.local_x), float(e.local_y)
        return None, None

    def actualizar_dibujo(self, cerrar_poligono=False):
        if self.img1_original is None: return
        
        img_dibujo = self.img1_original.copy()
        for i in range(len(self.puntos_reales)):
            cv2.circle(img_dibujo, self.puntos_reales[i], 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(img_dibujo, self.puntos_reales[i-1], self.puntos_reales[i], (0, 255, 255), 2)
                
        if cerrar_poligono and len(self.puntos_reales) > 2:
            cv2.line(img_dibujo, self.puntos_reales[-1], self.puntos_reales[0], (0, 255, 0), 3)

        self.img_recorte.src = self.convertir_cv2_a_base64(img_dibujo)
        self.page.update()

    def guardar_y_dibujar_punto(self, x_web, y_web):
        if self.img1_original is None: return

        alto_real, ancho_real = self.img1_original.shape[:2]
        x_real = int(x_web * ancho_real / self.ANCHO_VISOR_GRANDE)
        y_real = int(y_web * alto_real / self.ALTO_VISOR_GRANDE)

        if len(self.puntos_reales) > 0:
            ultimo_x, ultimo_y = self.puntos_reales[-1]
            distancia = ((x_real - ultimo_x)**2 + (y_real - ultimo_y)**2)**0.5
            if distancia < 15:
                return

        self.puntos_reales.append((x_real, y_real))
        self.btn_limpiar.disabled = False
        self.actualizar_dibujo(cerrar_poligono=False)

    def trazar_borde(self, e):
        x, y = self.extraer_coordenadas(e)
        if x is not None and y is not None:
            self.guardar_y_dibujar_punto(x, y)

    def clic_suelto_borde(self, e):
        x, y = self.extraer_coordenadas(e)
        if x is not None and y is not None:
            self.guardar_y_dibujar_punto(x, y)

    def soltar_raton(self, e):
        if len(self.puntos_reales) > 2:
            self.actualizar_dibujo(cerrar_poligono=True)
            self.btn_analizar.disabled = False
            self.page.update()

    # ==========================================
    # PROCESAMIENTO CLÍNICO Y MOTOR PLOTLY 3D
    # ==========================================
    def procesar_analisis(self, e):
        if len(self.puntos_reales) < 3: return

        # 1. Segmentación e IA K-Means
        img_recortada, mascara_roi = self.analizador_2d.aplicar_mascara_poligono(self.img1_original, self.puntos_reales)
        img_aislada = cv2.bitwise_and(self.img1_original, self.img1_original, mask=(mascara_roi*255).astype(np.uint8))
        
        _, segmentos = self.analizador_2d.segmentar_kmeans(img_recortada)
        tej_nec, tej_gra, tej_esf = self.analizador_2d.ordenar_tejidos(segmentos)
        
        # --- FIX DEFINITIVO DE COLOR ---
        # Intercambiamos de raíz las variables devueltas por tu motor para corregir la inversión
        tej_correct_nec = tej_gra
        tej_correct_gra = tej_nec
        
        mapa_colores = np.zeros_like(img_recortada)
        mapa_colores[cv2.cvtColor(tej_correct_nec, cv2.COLOR_BGR2GRAY) > 0] = [50, 50, 50]    # Gris (Necrosis)
        mapa_colores[cv2.cvtColor(tej_esf, cv2.COLOR_BGR2GRAY) > 0] = [0, 255, 255]   # Amarillo (Esfacelo)
        mapa_colores[cv2.cvtColor(tej_correct_gra, cv2.COLOR_BGR2GRAY) > 0] = [0, 0, 255]     # Rojo (Granulación)
        
        # Configurar el fondo fuera de la úlcera en Blanco Puro
        fondo_blanco = np.ones_like(img_recortada) * 255
        mascara_tres_canales = np.expand_dims(mascara_roi, axis=-1)
        mapa_colores = np.where(mascara_tres_canales == 1, mapa_colores, fondo_blanco)
        
        # 2. Reconstrucción Fotogramétrica 3D
        mascara_8u = (mascara_roi * 255).astype(np.uint8)
        nube_puntos, _ = self.motor_3d.calcular_3d(self.img1_bgr, self.img2_bgr, self.pts1, self.pts2, mascara_8u)

        # =======================================================
        # GENERACIÓN DE MALLA INTERACTIVA TEXTURIZADA (HTML NATIVO)
        # =======================================================
        fig = go.Figure()

        if nube_puntos is not None and len(nube_puntos) > 5:
            # Localizamos todos los píxeles de la imagen dentro de la úlcera delimitada
            y_indices, x_indices = np.where(mascara_roi > 0)
            
            # Submuestreo controlado (paso de 12px) para asegurar rotación fluida a 60fps en web
            salto_muestreo = 12
            x_densos = x_indices[::salto_muestreo]
            y_densos = y_indices[::salto_muestreo]

            if len(x_densos) > 3:
                # Extraemos las coordenadas X, Y, Z conocidas de la fotogrametría
                puntos_x_sift = nube_puntos[:, 0]
                puntos_y_sift = nube_puntos[:, 1]
                puntos_z_sift = nube_puntos[:, 2]

                # Interpolamos el relieve Z para la cuadrícula completa de la lesión
                z_densos = griddata(
                    points=np.column_stack((puntos_x_sift, puntos_y_sift)),
                    values=puntos_z_sift,
                    xi=(x_densos, y_densos),
                    method='nearest'
                )

                # Mapeamos la textura fotográfica real (RGB) sobre cada vértice de la malla
                colores_vertices = []
                for x_p, y_p in zip(x_densos, y_densos):
                    b, g, r = self.img1_original[y_p, x_p]
                    colores_vertices.append(f"rgb({r},{g},{b})")

                # Generamos la topología matemática de triángulos
                triangulacion = Delaunay(np.column_stack((x_densos, y_densos)))
                indices_triangulos = triangulacion.simplices

                # Construimos la estructura Mesh3D interactiva texturizada
                fig.add_trace(go.Mesh3d(
                    x=x_densos,
                    y=y_densos,
                    z=z_densos,
                    i=indices_triangulos[:, 0],
                    j=indices_triangulos[:, 1],
                    k=indices_triangulos[:, 2],
                    vertexcolor=colores_vertices,
                    flatshading=True
                ))
        else:
            # Si no hay textura suficiente para triangular
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="text",
                text=["Textura insuficiente para estimar profundidad Z"],
                textposition="top center"
            ))

        # Ajustes de cámara y proporciones reales para fines clínicos
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, autorange="reversed"), 
                zaxis=dict(visible=False),
                aspectmode='data' 
            )
        )

        # -------------------------------------------------------------
        # FIX VISOR 3D: Convertimos a string HTML completo y codificamos en Base64
        # -------------------------------------------------------------
        html_str = fig.to_html(include_plotlyjs="cdn", full_html=True)
        b64_html = base64.b64encode(html_str.encode("utf-8")).decode("utf-8")
        data_uri = f"data:text/html;base64,{b64_html}"
        
        # Inyectamos de forma segura la URL de datos en el WebView web nativo
        self.visor_res_3d.content = fvw.WebView(
            url=data_uri,
            expand=True
        )

        # 3. Renderizado y Actualización de Pantallas
        self.visor_res_recorte.src = self.convertir_cv2_a_base64(img_aislada)
        self.visor_res_segmentado.src = self.convertir_cv2_a_base64(mapa_colores)

        # Enviamos las variables corregidas al calculador de métricas para que los porcentajes no bailen
        resultados = self.analizador_2d.calcular_areas_y_porcentajes(mascara_roi, tej_correct_nec, tej_correct_gra, tej_esf, factor_escala=0.25)
        self.lbl_area.value = f"{resultados['areas']['total']:.2f} mm²"
        self.lbl_nec.value = f"{resultados['porcentajes']['necrotico']:.1f}%"
        self.lbl_esf.value = f"{resultados['porcentajes']['esfacelo']:.1f}%"
        self.lbl_gra.value = f"{resultados['porcentajes']['granulacion']:.1f}%"

        self.pantalla_recorte.visible = False
        self.pantalla_resultados.visible = True
        self.page.update()

    def volver_al_recorte(self, e):
        self.pantalla_resultados.visible = False
        self.pantalla_recorte.visible = True
        self.page.update()

    def cargar_video_click(self, e):
        ruta_test = "tu_video_de_prueba.mp4"
        if not os.path.exists(ruta_test):
            self.txt_placeholder_recorte.value = f"Error: No está '{ruta_test}'"
            self.txt_placeholder_recorte.color = "red"
            self.page.update()
            return

        try:
            img1, img2, self.pts1, self.pts2 = self.adquisicion.escoger_2_imagenes_desde_video(ruta_test)
            self.img1_bgr = img1
            self.img1_original = img1.copy() 
            self.img2_bgr = img2
            self.puntos_reales = []
            
            self.img_recorte.src = self.convertir_cv2_a_base64(img1)
            self.img_recorte.visible = True
            self.txt_placeholder_recorte.visible = False
        except Exception as ex:
            print(f"Error cargando vídeo: {ex}")
        self.page.update()

    def limpiar_recorte_click(self, e):
        self.puntos_reales = []
        if self.img1_original is not None:
            self.img_recorte.src = self.convertir_cv2_a_base64(self.img1_original)
        self.btn_analizar.disabled = True
        self.btn_limpiar.disabled = True
        self.page.update()

def main(page: ft.Page):
    NurseCareApp(page)

if __name__ == "__main__":
    ft.run(main, view=ft.AppView.WEB_BROWSER, port=8552)