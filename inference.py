"""Inference exclusively with the project's downloaded checkpoints."""
from pathlib import Path
from threading import RLock
from time import perf_counter
import io
import gc
import sys
import pandas as pd
from PIL import Image, ImageOps
from model_store import model_manifest, ensure_weights

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
COLUMNS = ['varietas', 'confidence', 'x1', 'y1', 'x2', 'y2']
_inference_lock = RLock()

def experiments():
    return [MODEL_DIR / name for name in sorted(model_manifest())]

def read_image(data):
    with Image.open(io.BytesIO(data)) as img:
        if img.width * img.height > 25_000_000:
            raise ValueError('Gambar terlalu besar. Gunakan gambar maksimal 25 megapiksel.')
        return ImageOps.exif_transpose(img).convert('RGB')

def load_detector(path):
    from ultralytics import YOLO, RTDETR
    if not Path(path).is_file():
        ensure_weights(Path(path).parent.parent.name)
    detector = RTDETR(path) if 'rtdetr' in str(path).lower() else YOLO(path)
    return detector, RLock()

def predict(detector, lock, image, confidence, iou, size, device):
    with lock:
        start = perf_counter()
        result = detector.predict(source=image, conf=confidence, iou=iou,
                                  imgsz=size, device=device, verbose=False)[0]
        elapsed = (perf_counter() - start) * 1000
        rows = []
        for box in result.boxes:
            coords = box.xyxy[0].cpu().tolist()
            rows.append([result.names[int(box.cls.item())], float(box.conf.item()), *coords])
        annotated = Image.fromarray(result.plot()[:, :, ::-1])
    return {'image': annotated, 'rows': pd.DataFrame(rows, columns=COLUMNS),
            'elapsed_ms': elapsed, 'inference_ms': result.speed.get('inference', 0),
            'names': list(result.names.values())}

def run_detector(path, image, confidence, iou, size, device):
    """Keep only the active model in memory, including across user sessions."""
    with _inference_lock:
        detector = None
        try:
            detector, lock = load_detector(path)
            return predict(detector, lock, image, confidence, iou, size, device)
        finally:
            del detector
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if sys.platform.startswith('linux'):
                # glibc may retain freed CPU tensor allocations after collection.
                import ctypes
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError, OSError):
                    pass

def png_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()
