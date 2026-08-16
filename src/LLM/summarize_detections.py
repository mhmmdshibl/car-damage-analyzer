import json
from pathlib import Path
from collections import defaultdict


def load_detections(json_path: str):
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"detections.json not found at: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def group_by_image(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det["image"]].append(det)
    return grouped


def summarize_image(image_path, dets):
    # Count per class
    counts = defaultdict(int)
    for d in dets:
        counts[d["class_name"]] += 1

    summary_lines = [f"Image: {image_path}"]
    summary_lines.append("Detected damage:")

    if not dets:
        summary_lines.append("  - No damage detected.")
    else:
        for cls_name, count in counts.items():
            summary_lines.append(f"  - {count} × {cls_name}")

    return "\n".join(summary_lines)


def build_llm_prompt_for_image(image_path, dets):
    """
    Build a prompt you can paste into ChatGPT (or use via API later).
    """
    prompt = {
        "image": image_path,
        "detections": dets,
        "instructions": (
            "You are an automotive damage assessment assistant.\n"
            "Given the detections above (damage type, confidence, bounding boxes), "
            "write a short, clear damage report for this car image.\n"
            "Mention which types of damage were found, roughly where on the car they are "
            "(e.g., front/side/back if possible), and give a brief recommendation.\n"
        ),
    }
    return prompt


def main():
    # Update json_path to use the absolute path
    json_path = str(Path(__file__).resolve().parent.parent.parent / "runs_infer/yolov8n_cardd/detections.json")
    detections = load_detections(json_path)
    grouped = group_by_image(detections)

    print("\n=== PER-IMAGE DAMAGE SUMMARY ===\n")
    for image_path, dets in grouped.items():
        print(summarize_image(image_path, dets))
        print()

    # Example: build an LLM prompt for the first image
    first_image, first_dets = next(iter(grouped.items()))
    prompt_payload = build_llm_prompt_for_image(first_image, first_dets)

    print("\n=== SAMPLE LLM PROMPT PAYLOAD (for ChatGPT or API) ===\n")
    print(json.dumps(prompt_payload, indent=4))


if __name__ == "__main__":
    main()
