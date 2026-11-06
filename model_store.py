"""Resolve and lazily download application checkpoints."""
import json
from pathlib import Path
from threading import RLock

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
_download_lock = RLock()

def model_manifest():
    items = json.loads((ROOT / 'drive-manifest.json').read_text(encoding='utf-8'))
    result = {}
    for item in items:
        parts = item['path'].replace('\\', '/').split('/')
        if len(parts) != 3 or parts[1:] != ['weights', 'best.pt']:
            continue
        if parts[0] in ('', '.', '..') or ':' in parts[0]:
            raise ValueError('Invalid model manifest path')
        result[parts[0]] = item['id']
    return result

def ensure_weights(name):
    import gdown
    manifest = model_manifest()
    if name not in manifest:
        raise ValueError(f'Unknown model: {name}')
    target = MODEL_DIR / name / 'weights' / 'best.pt'
    with _download_lock:
        if target.is_file() and target.stat().st_size:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.pt.part')
        try:
            result = gdown.download(id=manifest[name], output=str(temporary), quiet=True)
            if not result or not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError('Model download failed')
            with temporary.open('rb') as stream:
                if stream.read(2) != b'PK':
                    raise RuntimeError('Downloaded file is not a PyTorch checkpoint')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return target
