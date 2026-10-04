import os
import sys
import json
import time
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Add src/model to path
sys.path.insert(0, os.path.dirname(__file__))

# Configure UTF-8 safe stdout for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _orig_print(*args, **kwargs)

from architecture import GROGUArchitecture
from dataset import GROGUBimodalDataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'grogu_dataset')
)
SPLITS_DIR = os.path.join(BASE_DIR, 'splits')
TRAIN_CSV  = os.path.join(SPLITS_DIR, 'train.csv')
VAL_CSV    = os.path.join(SPLITS_DIR, 'val.csv')
TEST_CSV   = os.path.join(SPLITS_DIR, 'test.csv')

SAVE_PATH           = os.path.join(os.path.dirname(__file__), 'grogu_model.pth')
QUANTIZED_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'grogu_model_quantized.pt')
RESULTS_PATH        = os.path.join(os.path.dirname(__file__), 'training_results.json')

BATCH_SIZE   = 32
STAGE1_EPOCHS = 3    # Frozen backbone: train tabular MLP + fusion head
STAGE2_EPOCHS = 5    # Fine-tune MobileNetV2 blocks 14-18 + head
TOTAL_EPOCHS  = STAGE1_EPOCHS + STAGE2_EPOCHS

CLASS_NAMES  = ['genuine', 'font_tampered', 'metadata_tampered', 'both_tampered']


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, meta, labels in tqdm(loader, desc="  Train", leave=False):
        images, meta, labels = images.to(device), meta.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, meta)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, meta, labels in tqdm(loader, desc="  Eval", leave=False):
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            outputs = model(images, meta)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels


def main():
    print("=" * 65)
    print("GROGU 2-Stage Bimodal Forensics — 10K Dataset Fine-Tuning")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Datasets
    print("Loading datasets...")
    train_ds = GROGUBimodalDataset(TRAIN_CSV)
    val_ds   = GROGUBimodalDataset(VAL_CSV)
    test_ds  = GROGUBimodalDataset(TEST_CSV)
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Initialize Model with frozen visual backbone
    model = GROGUArchitecture(num_classes=4, meta_features=13, freeze_visual=True).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    # Loss function with light label smoothing for better compound generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # -------------------------------------------------------------
    # STAGE 1: Train Head & Metadata Branch
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"STAGE 1: Warmup Fusion Head & Tabular Branch ({STAGE1_EPOCHS} Epochs)")
    print("Visual backbone is FROZEN. Learning rate: 4e-4")
    print("-" * 65)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=4e-4, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    results = {"epochs": [], "test": {}}
    best_val_f1 = 0.0

    current_epoch = 1
    for epoch in range(1, STAGE1_EPOCHS + 1):
        t0 = time.time()
        print(f"Epoch {current_epoch}/{TOTAL_EPOCHS} (Stage 1 - {epoch}/{STAGE1_EPOCHS})")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc,   val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_f1)
        elapsed = time.time() - t0

        epoch_result = {
            "epoch": current_epoch,
            "stage": 1,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "train_f1":   round(train_f1,   4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "val_f1":     round(val_f1,     4),
            "time_sec":   round(elapsed, 1)
        }
        results["epochs"].append(epoch_result)
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Macro F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f} ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  * Best model saved (Val Macro F1: {val_f1:.4f})\n")
        else:
            print()
        current_epoch += 1

    # -------------------------------------------------------------
    # STAGE 2: Fine-Tune Top MobileNetV2 Blocks (14-18)
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print(f"STAGE 2: Fine-Tuning High-Level Visual Filters ({STAGE2_EPOCHS} Epochs)")
    print("Unfreezing MobileNetV2 blocks 14-18 with differential learning rates")
    print("Visual LR: 2.5e-5 | Fusion/Tabular LR: 1.5e-4")
    print("-" * 65)

    # Load best checkpoint from stage 1
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.set_visual_trainable(unfreeze=True, from_block=14)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters in Stage 2: {trainable_params:,}\n")

    # Differential parameter groups
    visual_params = [p for p in model.visual_model.features.parameters() if p.requires_grad]
    head_params = list(model.metadata_model.parameters()) + list(model.classifier.parameters())

    optimizer_ft = optim.AdamW([
        {'params': visual_params, 'lr': 2.5e-5, 'weight_decay': 1e-4},
        {'params': head_params,   'lr': 1.5e-4, 'weight_decay': 1e-4}
    ])
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=STAGE2_EPOCHS, eta_min=1e-6)

    for epoch in range(1, STAGE2_EPOCHS + 1):
        t0 = time.time()
        print(f"Epoch {current_epoch}/{TOTAL_EPOCHS} (Stage 2 - {epoch}/{STAGE2_EPOCHS})")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer_ft, device)
        val_loss,   val_acc,   val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler_ft.step()
        elapsed = time.time() - t0

        epoch_result = {
            "epoch": current_epoch,
            "stage": 2,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "train_f1":   round(train_f1,   4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "val_f1":     round(val_f1,     4),
            "time_sec":   round(elapsed, 1)
        }
        results["epochs"].append(epoch_result)
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | Macro F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}% | Macro F1: {val_f1:.4f} ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  * Best model saved (Val Macro F1: {val_f1:.4f})\n")
        else:
            print()
        current_epoch += 1

    # ─────────────────────────────────────────────
    # Final Test Evaluation
    # ─────────────────────────────────────────────
    print("=" * 65)
    print("Loading BEST model for final evaluation on 2,000 Test Documents...")
    print("=" * 65)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )
    report = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    print(f"\nFinal Test Accuracy : {test_acc*100:.2f}%")
    print(f"Final Test Macro F1 : {test_f1:.4f}")
    print("\nPer-Class Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0))

    results["test"] = {
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4),
        "per_class": report
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {RESULTS_PATH}")

    # ─────────────────────────────────────────────
    # Post-Training Quantization (INT8 Dynamic)
    # ─────────────────────────────────────────────
    print("\nApplying Dynamic INT8 Quantization for Edge Deployment...")
    try:
        model_cpu = GROGUArchitecture(num_classes=4, meta_features=13).to("cpu")
        model_cpu.load_state_dict(torch.load(SAVE_PATH, map_location="cpu"))
        model_cpu.eval()
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), QUANTIZED_SAVE_PATH)
        fp32_size = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        int8_size = os.path.getsize(QUANTIZED_SAVE_PATH) / (1024 * 1024)
        print(f"  FP32 Model Size: {fp32_size:.2f} MB")
        print(f"  INT8 Model Size: {int8_size:.2f} MB (~{int8_size/fp32_size*100:.1f}% footprint)")
    except Exception as e:
        print(f"  Quantization notice: {e}")

    # ─────────────────────────────────────────────
    # Regenerate Plot
    # ─────────────────────────────────────────────
    try:
        plot_script = os.path.join(os.path.dirname(__file__), '..', '..', 'plot_results.py')
        if os.path.exists(plot_script):
            subprocess.run([sys.executable, plot_script], check=False)
            print("Successfully updated results-graph.png")
    except Exception as e:
        print(f"Plot update notice: {e}")

    print("\n" + "=" * 65)
    print("GROGU TRAINING COMPLETE!")
    print("=" * 65)


if __name__ == '__main__':
    main()
