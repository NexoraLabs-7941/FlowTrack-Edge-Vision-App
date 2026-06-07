import cv2
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaProducer
from ultralytics import YOLO

app = FastAPI(title="FlowTrack Edge Vision Controller")

# Habilitar CORS para que tu Front en producción pueda enviar comandos sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción lo cambias por la URL de tu hosting
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StreamController:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.running = False
        self.cap = None
        self.ffmpeg_proc = None
        self.thread = None
        
        # Parámetros del Servidor de Streaming (MediaMTX)
        # En producción cambias 'localhost' por la IP pública de tu servidor central
        self.rtmp_url = "rtmp://localhost:1935/live/aforo_tienda"
        
        # Conexión a Kafka en Confluent Cloud
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['pkc-56d1g.eastus.azure.confluent.cloud:9092'],
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username='TU_API_KEY',
                sasl_plain_password='TU_API_SECRET',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("[INFO] Conectado a Confluent Cloud exitosamente")
        except Exception as e:
            print(f"[ERROR] Falló la conexión a Kafka: {e}")
            self.producer = None

    def _iniciar_ffmpeg(self, width, height):
        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f"{width}x{height}", '-r', '30',
            '-i', '-', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', '-f', 'flv',
            self.rtmp_url
        ]
        # Redirigimos stderr a devnull para no llenar la consola con logs técnicos de FFmpeg
        self.ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _worker_loop(self, camera_id):
        """Bucle que procesa los frames y los envía a MediaMTX"""
        # Intentamos abrir el ID de cámara que mandó el usuario (0, 1, o una ruta de archivo)
        try:
            # Si el usuario mandó un número como string "0", lo convertimos a entero
            cam_source = int(camera_id) if camera_id.isdigit() else camera_id
        except:
            cam_source = camera_id

        self.cap = cv2.VideoCapture(cam_source)
        success, frame = self.cap.read()
        
        if not success:
            print(f"[ERROR] No se pudo leer el origen de video: {camera_id}")
            self.running = False
            return

        height, width, _ = frame.shape
        self._iniciar_ffmpeg(width, height)
        
        last_counts = {}

        while self.cap.isOpened() and self.running:
            success, frame = self.cap.read()
            if not success:
                break

            # Inferencia optimizada de YOLOv8s (Clase 0 = Personas)
            results = self.model(frame, stream=True, conf=0.5, imgsz=320, classes=[0], verbose=False)
            
            local_counts = {}
            annotated_frame = frame.copy()
            
            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    local_counts["person"] = local_counts.get("person", 0) + 1

            # Lógica de Kafka (Lista para el futuro, no bloquea)
            if local_counts != last_counts:
                if self.producer:
                    evento = {
                        "dispositivo_id": f"camara_{camera_id}",
                        "timestamp": datetime.now().isoformat(),
                        "detecciones": local_counts if local_counts else {"person": 0}
                    }
                    try:
                        self.producer.send('flowtrack-detecciones-afluencia', value=evento)
                    except:
                        pass
                last_counts = local_counts.copy()

            # Empujar frame analizado a MediaMTX
            try:
                self.ffmpeg_proc.stdin.write(annotated_frame.tobytes())
            except:
                break

        # Limpieza al apagar la cámara
        if self.cap:
            self.cap.release()
        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.wait()
            except:
                pass
        print(f"[INFO] Streaming del origen {camera_id} finalizado limpiamente.")

    def start(self, camera_id):
        if self.running:
            return False # Ya hay una cámara transmitiendo
            
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, args=(camera_id,), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return True

# Instanciamos el controlador global
stream_manager = StreamController()

# -----------------------------------------------------------------
# 🌐 ENDPOINTS DE CONTROL PARA EL FRONTEND
# -----------------------------------------------------------------

@app.post("/api/v1/vision/start")
def iniciar_camara(camera_id: str = "0"):
    """El frontend llama aquí mandando el ID de la cámara que quiere activar (ej: '0' o '1')"""
    exito = stream_manager.start(camera_id)
    if not exito:
        raise HTTPException(status_code=400, detail="El streaming ya está activo o no se pudo iniciar.")
    
    # Le respondemos al front con la URL HLS que generará MediaMTX para que la incruste en su HTML
    return {
        "status": "success",
        "message": f"Cámara {camera_id} activada exitosamente.",
        "stream_url": "http://localhost:8888/live/aforo_tienda/index.m3u8"
    }

@app.post("/api/v1/vision/stop")
def detener_camara():
    """El frontend llama aquí cuando el usuario sale de la sección de aforo para apagar la cámara"""
    exito = stream_manager.stop()
    if not exito:
        raise HTTPException(status_code=400, detail="No hay ningún streaming activo para detener.")
    return {"status": "success", "message": "Streaming detenido correctamente."}

if __name__ == "__main__":
    import uvicorn
    
    # -----------------------------------------------------------------
    # 🌐 MODO DE CONTROL PURO POR ENDPOINTS (Listo para el Frontend)
    # -----------------------------------------------------------------
    print("\n" + "="*50)
    print("[STATUS] Microservicio Edge Vision de FlowTrack Iniciado.")
    print("[STATUS] Esperando órdenes del Frontend o Swagger...")
    print("[SERVER] Entra a probarlo aquí: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    # Arrancamos Uvicorn de forma normal. Ahora el código se quedará quieto 
    # y la cámara SOLO se encenderá cuando tú lo solicites en la web.
    uvicorn.run(app, host="0.0.0.0", port=8000)