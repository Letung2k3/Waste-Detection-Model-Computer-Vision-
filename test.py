from ultralytics import YOLO

MODEL_PATH = "training/runs/waste_detection_cpu/weights/best.pt"
SOURCE = "D:/ML_DC_DL/project_ml/datasets/images/test/biodegradable3_jpg.rf.818190ecb4ffa1e4dffdbe16b164bb30.jpg"  # hoặc 0 cho webcam


def predict():
    model = YOLO(MODEL_PATH)
    model.predict(
        source=SOURCE,
        imgsz=416,
        conf=0.4,
        device="cpu",
        save=True
    )


if __name__ == "__main__":
    predict()