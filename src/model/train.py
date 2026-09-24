import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm

# Add src/model to path
sys.path.insert(0, os.path.dirname(__file__))
from architecture import GROGUArchitecture
from dataset import GROGUBimodalDataset

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'grogu_dataset')
)
SPLITS_DIR = os.path.join(BASE_DIR, 'splits')
TRAIN_CSV  = os.path.join(SPLITS_DIR, 'train.csv')
VAL_CSV    = os.path.join(SPLITS_DIR, 'val.csv')
TEST_CSV   = os.path.join(SPLITS_DIR, 'test.csv')
SAVE_PATH  = os.path.join(os.path.dirname(__file__), 'grogu_model.pth')
RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'training_results.json')

BATCH_SIZE   = 16   # smaller batch on CPU to avoid OOM
EPOCHS       = 3    # quick run to get real results fast
LR           = 1e-4
WEIGHT_DECAY = 1e-5
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
    print("=" * 60)
    print("GROGU Bimodal Forensics — Training Run")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Datasets
    print("Loading datasets...")
    train_ds = GROGUBimodalDataset(TRAIN_CSV)
    val_ds   = GROGUBimodalDataset(VAL_CSV)
    test_ds  = GROGUBimodalDataset(TEST_CSV)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = GROGUArchitecture(num_classes=4, meta_features=13, freeze_visual=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} total params | {trainable_params:,} trainable (backbone frozen)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    results = {"epochs": [], "test": {}}
    best_val_f1 = 0.0

    print(f"\nTraining for {EPOCHS} epochs (backbone frozen)...\n")
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc,   val_f1, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc,  4),
            "train_f1":   round(train_f1,   4),
            "val_loss":   round(val_loss,   4),
            "val_acc":    round(val_acc,    4),
            "val_f1":     round(val_f1,     4),
        }
        results["epochs"].append(epoch_result)
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Macro F1: {train_f1:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f}   | Acc: {val_acc:.4f}   | Macro F1: {val_f1:.4f}\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  * Best model saved (Val Macro F1: {val_f1:.4f})\n")

    # ── Final Test Evaluation ──
    print("=" * 60)
    print("Loading best model for final TEST evaluation...")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc, test_f1, test_preds, test_labels = eval_epoch(
        model, test_loader, criterion, device
    )
    report = classification_report(
        test_labels, test_preds,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    print(f"\nTest Accuracy  : {test_acc:.4f}")
    print(f"Test Macro F1  : {test_f1:.4f}")
    print("\nPer-Class Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0))

    results["test"] = {
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4),
        "per_class": report
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
