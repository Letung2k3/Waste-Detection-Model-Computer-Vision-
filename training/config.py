
RESUME = False
TRAIN_CONFIG = {
    # ---------------- CORE ----------------
    "task": "detect",
    "mode": "train",
    "model": (
        "training/runs/waste_detection_v1/weights/last.pt"
        if RESUME else "yolov8n.pt"
    ),         # ✅ NHẸ NHẤT (BẮT BUỘC)
    "data": "../datasets/data.yaml",
    "epochs": 30,                   # 🔥 giảm từ 100 → 30
    "batch": 4,                     # 🔥 nhỏ để CPU chịu được
    "imgsz": 416,                   # 🔥 giảm từ 640 → 416
    "device": "cpu",

    # ---------------- PERFORMANCE ----------------
    "workers": 2,                   # 🔥 CPU thường chỉ nên 2
    "cache": False,                 # ❌ CPU yếu → tắt cache
    "amp": False,                   # ❌ chỉ cho GPU
    "pretrained": True,

    # ---------------- OPTIMIZER ----------------
    "optimizer": "SGD",             # 🔥 SGD nhẹ hơn Adam
    "lr0": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,

    # ---------------- VALIDATION ----------------
    "val": True,
    "save": True,
    "save_period": 5,               # 🔥 5 epoch lưu 1 lần
    "plots": True,

    # ---------------- LOGGING ----------------
    "project": "./runs",
    "name": "waste_detection_cpu",
    "exist_ok": True,
    "verbose": True,

    # ---------------- RESUME ----------------
    "resume": RESUME,                # bật khi train tiếp

    # ---------------- SPEED HACK ----------------
    "close_mosaic": 10,             # 🔥 giảm augmentation
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.5,
    "shear": 0.0,
    "fliplr": 0.5,
}

