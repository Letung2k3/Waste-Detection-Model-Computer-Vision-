from ultralytics import YOLO

MODEL_PATH = "runs/waste_detection_cpu/weights/best.pt"
DATA_PATH = "../data.yaml"


def evaluate():
    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_PATH,
        imgsz=416,
        device="cpu"
    )

    print("\n===== Evaluation Results =====")
    print(metrics)


if __name__ == "__main__":
    evaluate()