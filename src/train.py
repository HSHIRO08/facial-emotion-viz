"""
Script huấn luyện model Facial Expression Recognition (2D-CNN hoặc transfer learning).

Yêu cầu trước khi chạy: đã có train_manifest.csv và val_manifest.csv trong data/processed/
(xem build_manifest_template.py để tạo manifest từ data/raw/<label>/*.jpg).

Chạy: python train.py
"""

import os
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CLASS_LABELS,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAIN_MANIFEST,
    VAL_MANIFEST,
    WEIGHT_DECAY,
)
from dataset import build_dataloader
from model import build_model


def _compute_class_weights(manifest_path):
    """Trọng số tỷ lệ nghich với tần suất lớp, để CrossEntropyLoss đền bù class bị lệch
    (vd: surprise chiếm quá nhiều ảnh train sẽ bị phạt ít hơn, lớp hiếm được ưu tiên hơn)."""
    manifest = pd.read_csv(manifest_path)
    counts = manifest["label"].value_counts()
    weights = [len(manifest) / (len(CLASS_LABELS) * counts.get(label, 1)) for label in CLASS_LABELS]
    return torch.tensor(weights, dtype=torch.float32)


def _run_one_epoch(model, dataloader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_training:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def main():
    if not os.path.isfile(TRAIN_MANIFEST) or not os.path.isfile(VAL_MANIFEST):
        raise FileNotFoundError(
            "Không tìm thấy train_manifest.csv / val_manifest.csv trong data/processed/. "
            "Xem build_manifest_template.py để biết cách tạo manifest từ dataset gốc."
        )

    train_loader = build_dataloader(TRAIN_MANIFEST, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, augment=True)
    val_loader = build_dataloader(VAL_MANIFEST, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, augment=False)

    model = build_model().to(DEVICE)
    class_weights = _compute_class_weights(TRAIN_MANIFEST).to(DEVICE)
    print("Class weights (theo thứ tự CLASS_LABELS):", class_weights.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss, train_acc = _run_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_one_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)
        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch:03d}/{NUM_EPOCHS}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                best_checkpoint_path,
            )
            print(f"  -> Lưu checkpoint tốt nhất tại {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Dừng sớm sau {epoch} epoch (không cải thiện val_loss trong {EARLY_STOPPING_PATIENCE} epoch).")
                break

    print("Huấn luyện hoàn tất. Best val_loss:", best_val_loss)


if __name__ == "__main__":
    main()
