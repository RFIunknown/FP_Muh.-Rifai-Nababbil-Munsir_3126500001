"""
Final Project Oprec PENSHIP - Simplified & Optimized Script
Cukup jalankan: python train.py
"""

import csv
import gc
import os
import time
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ==================== CONFIGURATION ====================
ROBOFLOW_WORKSPACE = "rifais-workspace-lekje"
ROBOFLOW_PROJECT = "final-project-penship"
ROBOFLOW_VERSION = 12
DATASET_DIR = "dataset"

MODEL = "yolov8s.pt"
IMGSZ = 640
SEED = 0
WORKERS = 4
CACHE = "ram" 

EPOCHS = 30
BATCH = 16
PATIENCE = 10
RESULTS_CSV = "hasil_eksperimen.csv"
# =======================================================

def select_device():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU  : {props.name} | VRAM {props.total_memory / 1024**3:.1f} GB")
        return 0
    print("[!] GPU CUDA tidak terdeteksi, memakai CPU.")
    return "cpu"

def fix_data_yaml(data_yaml: Path) -> Path:
    root = data_yaml.parent.resolve()
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["path"] = str(root)
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        img_dir = root / folder / "images"
        if img_dir.exists():
            data[key] = str(img_dir)
        elif key == "test":
            data.pop("test", None)

    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return data_yaml

def get_dataset() -> Path:
    data_yaml = Path(DATASET_DIR) / "data.yaml"
    if data_yaml.exists():
        print(f"Dataset sudah ada, skip download: {data_yaml}")
        return data_yaml

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY belum di-set di CMD.")

    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    version = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT).version(ROBOFLOW_VERSION)
    dataset = version.download("yolov8", location=DATASET_DIR)
    return Path(dataset.location) / "data.yaml"

def main():
    start = time.time()
    device = select_device()
    data_path = fix_data_yaml(get_dataset())

    print(f"\n=== Memulai Training YOLOv8 (Epochs: {EPOCHS}, Batch: {BATCH}) ===")
    
    batch_size = BATCH
    save_dir = None
    
    # LOOP TRAINING DENGAN ANTI-VRAM HABIS (OOM)
    while True:
        try:
            model = YOLO(MODEL)
            model.train(
                data=str(data_path),
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=batch_size,
                seed=SEED,
                device=device,
                workers=WORKERS,
                cache=CACHE,
                patience=PATIENCE,
                cos_lr=True,
                warmup_epochs=3.0,
                close_mosaic=10,
                
                # --- AUGMENTASI AMAN (TANPA MIXUP & ERASING) ---
                hsv_v=0.5,
                degrees=10.0,
                scale=0.5,
                mosaic=0.5,
                fliplr=0.5,
                flipud=0.0
            )
            save_dir = Path(model.trainer.save_dir)
            break
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if batch_size <= 2:
                raise RuntimeError("VRAM tidak cukup walau batch=2.")
            batch_size //= 2
            print(f"\n[!] VRAM penuh. Mengulang otomatis dengan batch={batch_size} ...\n")

    # EVALUASI & PENCATATAN HASIL
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n=== Training Selesai! Model terbaik: {best_pt} ===")

    # Catat ke CSV untuk laporan README
    with open(data_path, "r", encoding="utf-8") as f:
        split = "test" if "test" in yaml.safe_load(f) else "val"
    
    model = YOLO(str(best_pt))
    m = model.val(data=str(data_path), imgsz=IMGSZ, split=split, batch=8, plots=False)
    minutes = (time.time() - start) / 60
    
    print(f"[{split}] mAP50: {m.box.map50:.3f} | Waktu: {minutes:.1f} menit")
    
    row = {
        "skenario": "Optimasi Akhir & Bebas Halusinasi",
        "epochs": EPOCHS,
        "batch": batch_size,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "menit": round(minutes, 1)
    }
    
    new_file = not Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file: w.writeheader()
        w.writerow(row)
        
    print(f"Hasil otomatis dicatat di {RESULTS_CSV}")

if __name__ == "__main__":
    main()