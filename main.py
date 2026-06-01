import cv2
import threading
import time
import json
from datetime import datetime
from kafka import KafkaProducer
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

        # ---------------------------------------------------------
        # [NUEVO KAFKA]: 2. Configuración del Productor de Kafka
        # ---------------------------------------------------------
        # Aquí pones la IP de tu servidor Kafka o la de Azure Event Hubs
        self.kafka_broker = 'localhost:9092' 
        self.topic_personas = 'bodega.eventos.personas'
        self.topic_restock = 'bodega.eventos.restock'
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_broker],
                # value_serializer convierte automáticamente tu diccionario de Python a un JSON válido
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("[KAFKA] Productor conectado al Broker exitosamente.")
        except Exception as e:
            print(f"[KAFKA ERROR] No se pudo conectar al Broker: {e}")
            self.producer = None # Fallback por si Kafka está apagado
        # ---------------------------------------------------------

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
                
                # OPTIMIZACIÓN 1: Procesar solo 1 de cada 3 frames
                if frame_counter % 3 == 0:
                    # OPTIMIZACIÓN 2: Reducir imgsz a 320
                    results = self.model(self.frame_actual, stream=True, conf=0.5, 
                                         imgsz=320, classes=self.clases_a_filtrar, verbose=False)
                    
                    local_counts = {}
                    for r in results:
                        self.results_plot = r.plot()
                        for box in r.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            local_counts[class_name] = local_counts.get(class_name, 0) + 1
                    
                    self.current_counts = local_counts
            else:
                time.sleep(0.01)

    # ---------------------------------------------------------
    # [NUEVO KAFKA]: 3. Método para empaquetar y enviar el evento
    # ---------------------------------------------------------
    def enviar_evento_kafka(self, topico, conteos):
        if self.producer is None:
            return # Si no hay conexión, ignoramos para no trabar la cámara
            
        # Armamos el payload (El JSON que viajará por la red)
        evento = {
            "dispositivo_id": "camara_bodega_lima_1",
            "timestamp": datetime.now().isoformat(),
            "modo": self.modo_actual,
            "detecciones": conteos if conteos else {"ninguno": 0}
        }
        
        # Enviar de forma asíncrona (no bloquea el video)
        self.producer.send(topico, value=evento)
        print(f"[KAFKA ENVÍO] -> Tópico: {topico} | Payload: {evento}")
    # ---------------------------------------------------------

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
            display_frame = self.results_plot if self.results_plot is not None else self.frame_actual

            if display_frame is not None:
                cv2.putText(display_frame, f"Modo: {self.modo_actual}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if self.modo_actual == "CONTAR_PERSONAS" else (0, 165, 255), 2)
                cv2.imshow("FlowTrack - Edge Ultra Speed", display_frame)

            # ---------------------------------------------------------
            # [NUEVO KAFKA]: 4. Disparador del Evento
            # ---------------------------------------------------------
            if self.current_counts != self.last_counts:
                # Si el número de personas o cajas cambió, enviamos el mensaje a Kafka
                print(f"[{self.modo_actual}]: {self.current_counts if self.current_counts else 'Despejado'}")
                
                # Decidimos a qué canal de Kafka enviarlo
                if self.modo_actual == "CONTAR_PERSONAS":
                    self.enviar_evento_kafka(self.topic_personas, self.current_counts)
                elif self.modo_actual == "RESTOCK":
                    self.enviar_evento_kafka(self.topic_restock, self.current_counts)

                self.last_counts = self.current_counts.copy()
            # ---------------------------------------------------------

            # Captura de teclado
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

        # Limpieza
        self.cap.release()
        cv2.destroyAllWindows()
        
        # [NUEVO KAFKA]: Asegurar que todos los mensajes pendientes se envíen antes de cerrar
        if self.producer is not None:
            self.producer.flush() 
            self.producer.close()

if __name__ == "__main__":
    detector = EdgeVisionSpeed()
    detector.run()