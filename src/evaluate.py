"""
Đánh giá model đã huấn luyện trên tập test_manifest.csv: accuracy, precision/recall/F1
(macro, phù hợp với dataset micro-expression thường mất cân bằng lớp), và confusion matrix.

Chạy: python evaluate.py [--checkpoint checkpoints/best_model.pt]
"""

import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from config import CHECKPOINT_DIR, CLASS_LABELS, DEVICE, NUM_WORKERS, TEST_MANIFEST
from dataset import build_dataloader
from model import build_model


def evaluate(checkpoint_path, manifest_path=TEST_MANIFEST, batch_size=8):
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Không tìm thấy manifest test: {manifest_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    dataloader = build_dataloader(manifest_path, batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())

    report = classification_report(all_labels, all_predictions, target_names=CLASS_LABELS, digits=4, zero_division=0)
    print(report)

    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(len(CLASS_LABELS))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Facial Expression Recognition")
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(checkpoint_path), "confusion_matrix.png")
    plt.savefig(output_path)
    print(f"Đã lưu confusion matrix tại: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    parser.add_argument("--manifest", default=TEST_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.manifest, args.batch_size)
