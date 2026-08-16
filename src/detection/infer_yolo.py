import sys
# Add sys.path print statement to debug Python environment
print("Python sys.path:", sys.path)

# Add the correct path to sys.path to resolve the ModuleNotFoundError
sys.path.append('/opt/anaconda3/lib/python3.12/site-packages')
print("Updated Python sys.path:", sys.path)

print(sys.path)
import argparse
from pathlib import Path
from ultralytics import YOLO
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO car-damage detection")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "models/cardd_yolov8n_best.pt"),
        help="Path to trained weights",
    )
    # Update the default source path to the correct directory for sample images
    parser.add_argument(
        "--source",
        type=str,
        default=str(Path(__file__).resolve().parent.parent.parent / "sample"),
        help="Image file or directory to run inference on",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = YOLO(str(weights_path))

    # Update the `project` argument to save results in the `Car Damage Analyzer/runs_infer` directory
    results = model.predict(
        source=args.source,  # image or folder
        imgsz=640,
        conf=args.conf,
        save=True,
        project=str(Path(__file__).resolve().parent.parent.parent / "runs_infer"),  # Save in `Car Damage Analyzer/runs_infer`
        name="yolov8n_cardd",
    )

    print(f"Inference complete. Annotated images saved in: "
          f"{Path('runs_infer/yolov8n_cardd').resolve()}")
    

    # Optional: print detections from first image
    if results:
        print("\nDetections for first image:")
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            print(f"  class_id={cls_id}, conf={conf:.2f}, xyxy={box.xyxy.tolist()}")

    # -------------------------------------------------
    # ✅ NEW: collect all detections into a JSON list
    # -------------------------------------------------
    detections = []

    # Define out_dir to align with the `project` argument in model.predict
    out_dir = Path(__file__).resolve().parent.parent.parent / "runs_infer/yolov8n_cardd"

    for result in results:
        image_path = str(result.path)
        for box in result.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "image": image_path,
                "class_id": cls_id,
                "class_name": model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox_xyxy": xyxy,
            })

    json_path = out_dir / "detections.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(detections, f, indent=4)

    print(f"\nSaved structured detections to: {json_path.resolve()}")


if __name__ == "__main__":
    main()
