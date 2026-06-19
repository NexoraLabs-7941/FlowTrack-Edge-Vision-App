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

# Habilitar CORS para que tu Front en desarrollo/producción envíe comandos sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def list_dshow_video_devices() -> list[str]:
    """Lista cámaras de video en Windows vía FFmpeg/DirectShow (mismo orden que OpenCV CAP_DSHOW)."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10
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


def _normalize_camera_label(label: str) -> str:
    text = label.lower()
    text = re.sub(r'laptop\s*[—\-]\s*', '', text)
    text = re.sub(r'celular\s*[—\-]\s*', '', text)
    text = re.sub(r'\([0-9a-f]{4}:[0-9a-f]{4}\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())


def resolve_camera_index(camera_id: str, camera_label: Optional[str] = None) -> int:
    """
    Resuelve el índice OpenCV correcto. El orden del navegador (MediaDevices) no coincide
    con DirectShow: Iriun suele ser índice 0 aunque en el browser aparezca segundo.
    """
    devices = list_dshow_video_devices()
    if devices:
        print(f"[INFO] Cámaras DirectShow detectadas: {devices}")

    if camera_label and devices:
        want_iriun = 'iriun' in camera_label.lower()
        label_tokens = [t for t in _normalize_camera_label(camera_label).split() if len(t) > 2]

        best_idx: Optional[int] = None
        best_score = -1
        for idx, name in enumerate(devices):
            norm_name = _normalize_camera_label(name)
            is_iriun = 'iriun' in norm_name
            if want_iriun and not is_iriun:
                continue
            if not want_iriun and is_iriun:
                continue

            score = sum(1 for token in label_tokens if token in norm_name)
            if norm_name in _normalize_camera_label(camera_label):
                score += 10
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score > 0:
            print(f"[INFO] Cámara resuelta por nombre '{camera_label}' -> '{devices[best_idx]}' (índice {best_idx})")
            return best_idx

    if camera_id.isdigit():
        idx = int(camera_id)
        if devices and 0 <= idx < len(devices):
            print(f"[INFO] Cámara por índice {idx}: '{devices[idx]}'")
        return idx

    return 0

class StreamController:
    def __init__(self):
        # Carga el modelo YOLOv8s. Si no existe en la carpeta, se descargará automáticamente
        self.model = YOLO('yolov8s.pt')
        self.running = False
        self.cap = None
        self.ffmpeg_proc = None
        self.thread = None
        
        # Parámetros del Servidor de Streaming (MediaMTX)
        self.rtmp_url = "rtmp://localhost:1935/live/aforo_tienda"
        
        # Conexión a Kafka en Confluent Cloud
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
            print("[INFO] Conectado a Confluent Cloud exitosamente")
        except Exception as e:
            print(f"[ERROR] Falló la conexión a Kafka: {e}")
            self.producer = None
            print("[WARN] No se pudo conectar a Kafka. El sistema operará en modo local sin telemetría.")

    def _iniciar_ffmpeg(self, width, height):
        # 🔥 SOLUCIÓN CRUCIAL: Aseguramos que el ancho y alto se fuercen a números pares en el encoder
        # El filtro scale=-2:trunc(ih/2)*2 corrige cualquier dimensión impar automáticamente
        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', f"{width}x{height}", '-r', '30',
            '-i', '-', 
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', # Fuerza resoluciones pares obligatorias para x264
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast', 
            '-tune', 'zerolatency', # Optimización clave para streaming en tiempo real
            '-f', 'flv',
            self.rtmp_url
        ]
        self.ffmpeg_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _worker_loop(self, cam_index: int, camera_name: str):
        """Bucle optimizado y temporizado que procesa frames y los inyecta a MediaMTX"""
        cam_source = cam_index

        # Inicialización de la captura usando DirectShow en Windows para agilizar webcams
        self.cap = cv2.VideoCapture(cam_source, cv2.CAP_DSHOW)
        
        # Calentamiento inicial del hardware / búfer de lectura
        for _ in range(5):
            self.cap.read()
            time.sleep(0.05)

        success, frame = self.cap.read()
        
        if not success:
            print(f"[ERROR] No se pudo leer la cámara '{camera_name}' (índice {cam_index})")
            self.running = False
            return

        height, width, _ = frame.shape
        self._iniciar_ffmpeg(width, height)
        
        last_counts = {}

        # Reducir el búfer interno de OpenCV al mínimo para mitigar el lag acumulado
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        print(f"[INFO] Procesamiento YOLO activo en '{camera_name}' (índice OpenCV {cam_index})")

        while self.cap.isOpened() and self.running:
            start_time = time.time()  # ⏱️ Capturamos el milisegundo exacto de inicio del frame
            
            success, frame = self.cap.read()
            if not success:
                break

            # Inferencia optimizada reduciendo el tamaño de análisis (imgsz=320) enfocado solo en personas (class 0)
            results = self.model(frame, stream=True, conf=0.5, imgsz=320, classes=[0], verbose=False)
            
            local_counts = {}
            annotated_frame = frame.copy()
            
            for r in results:
                annotated_frame = r.plot()
                for box in r.boxes:
                    local_counts["person"] = local_counts.get("person", 0) + 1

            # Despacho asíncrono de telemetría hacia el broker de Kafka si el conteo cambia
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

            # Transferencia binaria del frame procesado hacia FFmpeg
            try:
                self.ffmpeg_proc.stdin.write(annotated_frame.tobytes())
            except:
                print("[WARN] Tubería de FFmpeg rota de forma inesperada.")
                break
            time.sleep(0.03)

            # 🚀 SINCRONIZACIÓN MAESTRA A 30 FPS: Evita la saturación del búfer de Windows
            time_elapsed = time.time() - start_time
            time_to_wait = (1.0 / 30.0) - time_elapsed
            if time_to_wait > 0:
                time.sleep(time_to_wait)

        if self.cap:
            self.cap.release()
            
        if self.ffmpeg_proc:
            try:
                self.ffmpeg_proc.stdin.close()
                self.ffmpeg_proc.terminate()  # 🔥 Le pide amablemente a FFmpeg que muera
                self.ffmpeg_proc.wait(timeout=2) # Le da 2 segundos de cortesía
            except:
                self.ffmpeg_proc.kill() # 💥 Si se resiste, lo aniquila del sistema operativo de raíz
        
        self.running = False
        print(f"[INFO] Streaming de '{camera_name}' finalizado y puertos liberados.")

        
    def start(self, cam_index: int, camera_name: str):
        if self.running:
            return False # Ya hay una cámara transmitiendo
        
        ##time.sleep(2)

        self.running = True
        self.thread = threading.Thread(
            target=self._worker_loop, args=(cam_index, camera_name), daemon=True
        )
        self.thread.start()
        return True

    def stop(self):
        if not self.running:
            return False
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        #return True
    
        if self.cap:
            self.cap.release()
            self.cap = None
        return True

# Instanciación global del gestor de streaming
stream_manager = StreamController()

# -----------------------------------------------------------------
# 🌐 ENDPOINTS DE CONTROL API (FastAPI)
# -----------------------------------------------------------------

@app.get("/api/v1/vision/cameras")
def listar_camaras():
    """Lista cámaras disponibles en el host (DirectShow) con su índice OpenCV."""
    devices = list_dshow_video_devices()
    return {
        "cameras": [
            {"index": idx, "name": name, "iriun": "iriun" in name.lower()}
            for idx, name in enumerate(devices)
        ]
    }

@app.post("/api/v1/vision/start")
def iniciar_camara(camera_id: str = "0", camera_label: Optional[str] = None):
    """Acción invocada por el orquestador Java para inicializar el aforo perimetral"""
    cam_index = resolve_camera_index(camera_id, camera_label)
    devices = list_dshow_video_devices()
    camera_name = devices[cam_index] if devices and cam_index < len(devices) else (camera_label or f"camera_{cam_index}")

    stream_manager.start(cam_index, camera_name)
    
    return {
        "status": "success",
        "message": f"Cámara '{camera_name}' activada (índice OpenCV {cam_index}).",
        "stream_url": "http://localhost:8888/live/aforo_tienda/index.m3u8",
        "camera_index": cam_index,
        "camera_name": camera_name
    }

@app.post("/api/v1/vision/stop")
def detener_camara():
    """Acción invocada para desmantelar y apagar el flujo de streaming activo"""
    exito = stream_manager.stop()
    if not exito:
        raise HTTPException(status_code=400, detail="No hay ningún streaming activo para detener.")
    return {"status": "success", "message": "Streaming detenido correctamente."}


def _run_inventory_detection(frame: np.ndarray) -> dict:
    """
    Ejecuta YOLO sobre un frame/imagen y cuenta objetos detectados.
    Ignora estrictamente a las personas (clase 0) para el módulo de Restock.
    """
    names = stream_manager.model.names
    
    # Seleccionamos las clases típicas de productos (ej: botella es la clase 39 en COCO)
    # Si usas clases personalizadas, puedes listarlas aquí directamente.
    clases_inventario = [int(cls_id) for cls_id in names.keys() if int(cls_id) != 0]

    # Ejecutamos inferencia aislando las clases de inventario a nivel de tensores
    results = stream_manager.model(
        frame, 
        conf=0.45,       # Subimos ligeramente la confianza para mitigar falsos positivos en el aula
        imgsz=640, 
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
            
            # 🛑 VALIDACIÓN AGRESIVA: Si por alguna razón OpenCV o YOLO cruzan la clase 0, la saltamos
            if cls_id == 0:
                continue
                
            class_name = result.names.get(cls_id, f"item_{cls_id}")
            
            # Traducimos la clase común 'bottle' al nombre comercial de tu negocio si es necesario
            if class_name == 'bottle':
                class_name = 'Yogurt'  # Mapeo dinámico para que coincida con tu tabla de Angular

            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(float(box.conf[0]))
            
            # 🎨 Pintamos el recuadro verde de FlowTrack únicamente en los ítems válidos
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (34, 197, 94), 2)
            cv2.putText(
                annotated_frame, 
                f"{class_name} {float(box.conf[0]):.2f}", 
                (xyxy[0], xyxy[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (34, 197, 94), 2
            )

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
    """Detecta y cuenta objetos en una imagen subida (reposición de inventario)."""
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
    
    print("\n" + "="*50)
    print("[STATUS] Microservicio Edge Vision de FlowTrack Iniciado.")
    print("[STATUS] Esperando órdenes del Frontend o Swagger...")
    print("[SERVER] Entra a probarlo aquí: http://localhost:8000/docs")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)