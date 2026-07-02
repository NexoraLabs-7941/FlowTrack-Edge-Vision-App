import cv2
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from typing import Optional
import base64
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaProducer
from ultralytics import YOLO

app = FastAPI(title="FlowTrack Edge Vision Controller")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def list_dshow_video_devices() -> list[str]:
    try:
        result = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5
        )
    except Exception as exc:
        print(f"[WARN] No se pudieron enumerar cámaras DirectShow: {exc}")
        return []
    devices: list[str] = []
    for line in (result.stderr or '').splitlines():
        match = re.search(r'"([^"]+)" \(video\)', line)
        if match:
            devices.append(match.group(1))
    return devices

def resolve_camera_index(camera_id: str, camera_label: Optional[str] = None) -> int:
    devices = list_dshow_video_devices()
    print(f"\n[INFO] Cámaras detectadas: {devices}")
    label_lower = (camera_label or "").lower()
    is_iriun_request = 'iriun' in label_lower
    if devices:
        for idx, name in enumerate(devices):
            name_lower = name.lower()
            if not is_iriun_request:
                if 'iriun' not in name_lower and 'virtual' not in name_lower and 'obs' not in name_lower:
                    return idx
            else:
                if 'iriun' in name_lower:
                    return idx
    return int(camera_id) if camera_id.isdigit() else 0

class StreamController:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.running = False
        self.cap = None
        self.ffmpeg_proc = None
        self.thread = None
        self.log_file = None
        
        # Calentamiento en frío para inferencia rápida
        dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy_frame, verbose=False)
        self.rtmp_url = "rtmp://localhost:1935/live/aforo_tienda"
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=['pkc-56d1g.eastus.azure.confluent.cloud:9092'],
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username='FYTHUU7K3L2N43XK',
                sasl_plain_password='cflthuqy3igT8XQyZvyhpmuUO49okhvUbdbqorZQB9NN4gCtR0oPAHYKf+ClOj7w',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("[STATUS] Conexión establecida con Kafka de forma exitosa.")
        except:
            self.producer = None
            print("[WARN] No se pudo conectar a Kafka. Operando localmente.")

    def _iniciar_ffmpeg(self, width, height):
        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f"{width}x{height}", 
            '-framerate', '30', '-i', '-', 
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-r', '30', '-f', 'flv', self.rtmp_url
        ]
        self.log_file = open("ffmpeg_error.log", "w")
        self.ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log_file)

    def _worker_loop(self, cam_index: int, camera_name: str):
        cam_source = cam_index
        self.cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
        
        for _ in range(5):
            self.cap.read()
            time.sleep(0.05)

        success, frame = self.cap.read()
        if not success:
            print(f"[ERROR] No se pudo leer la cámara '{camera_name}'")
            self.running = False
            return

        height, width, _ = frame.shape
        self._iniciar_ffmpeg(width, height)
        last_counts = {}
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.cap.isOpened() and self.running:
            start_time = time.time() 
            success, frame = self.cap.read()
            if not success:
                break

            results = self.model(frame, stream=True, conf=0.5, imgsz=320, classes=[0], verbose=False)
            
            local_counts = {}
            annotated_frame = frame.copy()
            
            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    local_counts["person"] = local_counts.get("person", 0) + 1

            if local_counts != last_counts:
                if self.producer:
                    evento = {
                        "dispositivo_id": f"camara_{cam_index}",
                        "timestamp": datetime.now().isoformat(),
                        "detecciones": local_counts if local_counts else {"person": 0}
                    }
                    try:
                        self.producer.send('flowtrack-detecciones-afluencia', value=evento)
                    except:
                        pass
                last_counts = local_counts.copy()

            try:
                self.ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                self.ffmpeg_proc.stdin.flush() 
            except Exception as e:
                break
                
            time_elapsed = time.time() - start_time
            time_to_wait = (1.0 / 30.0) - time_elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        if self.cap: self.cap.release()
        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.terminate() 
                self.ffmpeg_proc.wait(timeout=2)
            except:
                self.ffmpeg_proc.kill() 
            finally:
                if self.log_file and not self.log_file.closed:
                    self.log_file.close()
        self.running = False

    def start(self, cam_index: int, camera_name: str):
        if self.running: return False
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, args=(cam_index, camera_name), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=5)
        if self.cap: self.cap.release()
        return True

stream_manager = StreamController()

# -----------------------------------------------------------------
# 🌐 ENDPOINTS DE CONTROL API (FastAPI)
# -----------------------------------------------------------------

# 🔥 RESTAURADO: El endpoint vital para que Angular liste las cámaras
@app.get("/api/v1/vision/cameras")
def listar_camaras():
    devices = list_dshow_video_devices()
    return {
        "cameras": [
            {"index": idx, "name": name, "iriun": "iriun" in name.lower()}
            for idx, name in enumerate(devices)
        ]
    }

@app.post("/api/v1/vision/start")
def iniciar_camara(camera_id: str = "0", camera_label: Optional[str] = None):
    idx = resolve_camera_index(camera_id, camera_label)
    camera_name = camera_label or f"camera_{idx}"
    
    stream_manager.start(idx, camera_name)
    
    return {
        "status": "success",
        "message": f"Cámara '{camera_name}' activada.",
        "stream_url": "http://localhost:8888/live/aforo_tienda/index.m3u8",
        "camera_index": idx,
        "camera_name": camera_name
    }

@app.post("/api/v1/vision/stop")
def detener_camara():
    exito = stream_manager.stop()
    if not exito:
        raise HTTPException(status_code=400, detail="No hay streaming activo.")
    return {"status": "success", "message": "Streaming detenido correctamente."}

def _run_inventory_detection(frame: np.ndarray) -> dict:
    """
    Ejecuta YOLO y filtra estrictamente clases de inventario.
    """
    # Clases permitidas: botellas, vasos, tazones y comida
    clases_inventario = [39, 41, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]

    results = stream_manager.model(
        frame, 
        conf=0.45,       
        imgsz=416, 
        classes=clases_inventario, 
        verbose=False
    )

    class_counts: dict[str, int] = {}
    confidences: list[float] = []
    annotated_frame = frame.copy()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = result.names.get(cls_id, f"item_{cls_id}")
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(float(box.conf[0]))
            
            # RESTAURADO: Pintado explícito del recuadro verde
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (34, 197, 94), 2)
            cv2.putText(
                annotated_frame, 
                f"{class_name} {float(box.conf[0]):.2f}", 
                (xyxy[0], xyxy[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2
            )

    # RESTAURADO: Variables críticas para el Frontend
    total_count = sum(class_counts.values())
    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "success",
        "total_count": total_count,
        "detections": class_counts,
        "confidence": avg_confidence,
        "annotated_image_base64": annotated_b64
    }

@app.post("/api/v1/vision/detect-image")
async def detectar_imagen(image: UploadFile = File(...)):
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="La imagen está vacía.")
    
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
        
    return _run_inventory_detection(frame)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)