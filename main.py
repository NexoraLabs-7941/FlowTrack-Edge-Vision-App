import cv2
import threading
import time
from ultralytics import YOLO

class EdgeVisionSpeed:
    def __init__(self):
        # 1. Cargamos el modelo
        self.model = YOLO('yolov8s.pt')
        
        # Configuración de clases
        self.CLASE_PERSONA = 0
        self.CLASES_PRODUCTOS = [39, 41, 64, 67]
        
        # Estados dinámicos
        self.modo_actual = "CONTAR_PERSONAS"
        self.clases_a_filtrar = [self.CLASE_PERSONA]
        
        # Hilos y Compartición de Video
        self.cap = cv2.VideoCapture(0)
        self.frame_actual = None
        self.running = True
        
        # Resultados compartidos
        self.results_plot = None
        self.current_counts = {}
        self.last_counts = {}

    def start_camera_thread(self):
        """Hilo dedicado exclusivamente a leer la cámara a máxima velocidad"""
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_actual = frame
            else:
                time.sleep(0.01)

    def start_inference_thread(self):
        """Hilo dedicado a la IA con salto de frames y resolución optimizada"""
        frame_counter = 0
        while self.running:
            if self.frame_actual is not None:
                frame_counter += 1
                
                # OPTIMIZACIÓN 1: Procesar solo 1 de cada 3 frames (Salto de frames)
                if frame_counter % 3 == 0:
                    # OPTIMIZACIÓN 2: Reducir imgsz a 320 para duplicar la velocidad en CPU
                    results = self.model(self.frame_actual, stream=True, conf=0.5, 
                                         imgsz=320, classes=self.clases_a_filtrar, verbose=False)
                    
                    local_counts = {}
                    for r in results:
                        self.results_plot = r.plot() # Dibujo asíncrono
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            local_counts[class_name] = local_counts.get(class_name, 0) + 1
                    
                    self.current_counts = local_counts
            else:
                time.sleep(0.01)

    def run(self):
        if not self.cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            return

        # Arrancar hilos secundarios de soporte
        threading.Thread(target=self.start_camera_thread, daemon=True).start()
        threading.Thread(target=self.start_inference_thread, daemon=True).start()

        print("\n=== SYSTEM ONLINE (MODO ULTRA FAST) ===")
        print("[CONTROLES] 'p': Personas | 'r': Restock | 'q': Salir\n")

        while self.running:
            # Si la IA ya procesó un frame con recuadros, usamos ese; si no, el frame limpio de la cámara
            display_frame = self.results_plot if self.results_plot is not None else self.frame_actual

            if display_frame is not None:
                # Mostrar el modo actual en pantalla
                cv2.putText(display_frame, f"Modo: {self.modo_actual}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if self.modo_actual == "CONTAR_PERSONAS" else (0, 165, 255), 2)
                
                cv2.imshow("FlowTrack - Edge Ultra Speed", display_frame)

            if self.current_counts != self.last_counts:
                print(f"[{self.modo_actual}]: {self.current_counts if self.current_counts else 'Despejado'}")
                self.last_counts = self.current_counts.copy()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('p') and self.modo_actual != "CONTAR_PERSONAS":
                self.modo_actual = "CONTAR_PERSONAS"
                self.clases_a_filtrar = [self.CLASE_PERSONA]
                self.results_plot = None
                print(f"\n[MODO] -> {self.modo_actual}")
            elif key == ord('r') and self.modo_actual != "RESTOCK":
                self.modo_actual = "RESTOCK"
                self.clases_a_filtrar = self.CLASES_PRODUCTOS
                self.results_plot = None
                print(f"\n[MODO] -> {self.modo_actual}")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EdgeVisionSpeed()
    detector.run()