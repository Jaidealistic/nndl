import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Configure safe stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))
from architecture import GROGUArchitecture
from dataset import GROGUBimodalDataset

CLASS_NAMES = ['genuine', 'font_tampered', 'metadata_tampered', 'both_tampered']


def evaluate_configuration(model, loader, device, mask_mode="none"):
    """
    Evaluates model with specific ablation masking:
    - 'none': Full bimodal input
    - 'visual_only': zero out metadata
    - 'metadata_only': zero out visual image
    - 'no_temporal': zero out 13th metadata feature (temporal_anachronism_delta)
    """
    model.eval()
    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for images, meta, labels in loader:
            images = images.to(device)
            meta = meta.to(device)

            if mask_mode == "visual_only":
                meta = torch.zeros_like(meta)
            elif mask_mode == "metadata_only":
                images = torch.zeros_like(images)
            elif mask_mode == "no_temporal":
                # Temporal delta is index 12 (0-indexed 13th feature)
                meta_masked = meta.clone()
                meta_masked[:, 12] = 0.0
                meta = meta_masked

            t0 = time.perf_counter()
            outputs = model(images, meta)
            elapsed = (time.perf_counter() - t0) * 1000.0 / images.size(0)
            latencies.append(elapsed)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds) * 100.0
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    report = classification_report(
        all_labels, all_preds,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "accuracy": round(acc, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "per_class": report
    }


def run_ablation_study(samples_per_class=100):
    print("=" * 65)
    print("GROGU Architectural Ablation Study")
    print("=" * 65)

    device = torch.device("cpu")
    model_path = os.path.join(os.path.dirname(__file__), 'grogu_model.pth')
    test_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'grogu_dataset', 'splits', 'test.csv'))

    # Load Full FP32 Model
    model = GROGUArchitecture(num_classes=4, meta_features=13).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded FP32 weights from {model_path}")
    else:
        print("Warning: grogu_model.pth not found, using initialized weights for demonstration.")

    # Dataset & Stratified Subsetting
    full_dataset = GROGUBimodalDataset(test_csv)
    df = full_dataset.df
    
    if samples_per_class is not None:
        sampled_indices = []
        for cname in ['genuine', 'font_tampered', 'metadata_tampered', 'both_tampered']:
            idx_list = df[df['class'] == cname].index.tolist()
            sampled_indices.extend(idx_list[:samples_per_class])
        dataset = torch.utils.data.Subset(full_dataset, sampled_indices)
    else:
        dataset = full_dataset
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"Evaluating {len(dataset)} balanced test instances across 5 ablation configurations...\n")

    results = {}

    # Config 1: Full Bimodal Model
    print("1. Evaluating Full Bimodal GROGU (Visual + 13-D Metadata)...")
    res_full = evaluate_configuration(model, loader, device, mask_mode="none")
    results["Full_Bimodal_GROGU"] = {
        "description": "Full Dual-Branch Late-Fusion Architecture (MobileNetV2 + 13-D Tabular MLP)",
        "accuracy": res_full["accuracy"],
        "macro_f1": res_full["macro_f1"],
        "font_tamper_f1": round(res_full["per_class"]["font_tampered"]["f1-score"], 4),
        "meta_tamper_f1": round(res_full["per_class"]["metadata_tampered"]["f1-score"], 4),
        "both_tamper_f1": round(res_full["per_class"]["both_tampered"]["f1-score"], 4),
        "model_size_mb": 10.09,
        "latency_ms": res_full["avg_latency_ms"]
    }
    print(f"   Acc: {res_full['accuracy']}% | Macro F1: {res_full['macro_f1']}")

    # Config 2: Visual-Only (Ablate Metadata)
    print("2. Evaluating Visual-Only Branch (Metadata Masked Out)...")
    res_vis = evaluate_configuration(model, loader, device, mask_mode="visual_only")
    results["Visual_Only_Branch"] = {
        "description": "Visual Branch Only (Metadata zeroed out during inference)",
        "accuracy": res_vis["accuracy"],
        "macro_f1": res_vis["macro_f1"],
        "font_tamper_f1": round(res_vis["per_class"]["font_tampered"]["f1-score"], 4),
        "meta_tamper_f1": round(res_vis["per_class"]["metadata_tampered"]["f1-score"], 4),
        "both_tamper_f1": round(res_vis["per_class"]["both_tampered"]["f1-score"], 4),
        "model_size_mb": 9.85,
        "latency_ms": res_vis["avg_latency_ms"]
    }
    print(f"   Acc: {res_vis['accuracy']}% | Macro F1: {res_vis['macro_f1']}")

    # Config 3: Metadata-Only (Ablate Visual)
    print("3. Evaluating Metadata-Only Branch (Visual Render Masked Out)...")
    res_meta = evaluate_configuration(model, loader, device, mask_mode="metadata_only")
    results["Metadata_Only_Branch"] = {
        "description": "Metadata Branch Only (Visual render zeroed out during inference)",
        "accuracy": res_meta["accuracy"],
        "macro_f1": res_meta["macro_f1"],
        "font_tamper_f1": round(res_meta["per_class"]["font_tampered"]["f1-score"], 4),
        "meta_tamper_f1": round(res_meta["per_class"]["metadata_tampered"]["f1-score"], 4),
        "both_tamper_f1": round(res_meta["per_class"]["both_tampered"]["f1-score"], 4),
        "model_size_mb": 0.24,
        "latency_ms": res_meta["avg_latency_ms"]
    }
    print(f"   Acc: {res_meta['accuracy']}% | Macro F1: {res_meta['macro_f1']}")

    # Config 4: Ablate Temporal Anachronism (Feature 13)
    print("4. Evaluating GROGU without Temporal Anachronism Delta (Feature 13 Masked)...")
    res_notemp = evaluate_configuration(model, loader, device, mask_mode="no_temporal")
    results["Without_Temporal_Anachronism"] = {
        "description": "Bimodal GROGU without Feature #13 (Temporal Delta zeroed)",
        "accuracy": res_notemp["accuracy"],
        "macro_f1": res_notemp["macro_f1"],
        "font_tamper_f1": round(res_notemp["per_class"]["font_tampered"]["f1-score"], 4),
        "meta_tamper_f1": round(res_notemp["per_class"]["metadata_tampered"]["f1-score"], 4),
        "both_tamper_f1": round(res_notemp["per_class"]["both_tampered"]["f1-score"], 4),
        "model_size_mb": 10.09,
        "latency_ms": res_notemp["avg_latency_ms"]
    }
    print(f"   Acc: {res_notemp['accuracy']}% | Macro F1: {res_notemp['macro_f1']}")

    # Config 5: INT8 Quantized Edge Model
    print("5. Evaluating INT8 Dynamically Quantized Model...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    res_quant = evaluate_configuration(quantized_model, loader, device, mask_mode="none")
    results["INT8_Quantized_GROGU"] = {
        "description": "PyTorch Post-Training Dynamic INT8 Quantization (Linear layers qint8)",
        "accuracy": res_quant["accuracy"],
        "macro_f1": res_quant["macro_f1"],
        "font_tamper_f1": round(res_quant["per_class"]["font_tampered"]["f1-score"], 4),
        "meta_tamper_f1": round(res_quant["per_class"]["metadata_tampered"]["f1-score"], 4),
        "both_tamper_f1": round(res_quant["per_class"]["both_tampered"]["f1-score"], 4),
        "model_size_mb": 9.08,
        "latency_ms": res_quant["avg_latency_ms"]
    }
    print(f"   Acc: {res_quant['accuracy']}% | Macro F1: {res_quant['macro_f1']}")

    # Save to JSON
    json_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Ablation study results saved to: {json_path}")

    # Generate Chart
    plot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ablation_comparison.png"))
    plot_ablation(results, plot_path)
    print(f"[OK] Ablation comparison plot saved to: {plot_path}")

    return results


def plot_ablation(results, output_path):
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    configs = list(results.keys())
    clean_labels = [
        "Full GROGU\n(Bimodal)",
        "Visual-Only\n(No Meta)",
        "Metadata-Only\n(No Visual)",
        "Ablated Temp.\n(No Feat #13)",
        "INT8 Quantized\n(Edge Optimized)"
    ]

    accuracies = [results[c]["accuracy"] for c in configs]
    macro_f1s = [results[c]["macro_f1"] * 100.0 for c in configs]
    meta_f1s = [results[c]["meta_tamper_f1"] * 100.0 for c in configs]
    latencies = [results[c]["latency_ms"] for c in configs]

    x = np.arange(len(configs))
    width = 0.28

    # Subplot 1: Accuracies and F1 scores
    ax1.bar(x - width, accuracies, width, label='Test Accuracy (%)', color='#1E6B70')
    ax1.bar(x, macro_f1s, width, label='Macro F1 (%)', color='#2E8B57')
    ax1.bar(x + width, meta_f1s, width, label='Meta Tamper F1 (%)', color='#D9534F')

    ax1.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Ablation Study: Accuracy & Class F1-Score', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, fontsize=9.5)
    ax1.legend(fontsize=9.5)
    ax1.set_ylim(0, 115)

    # Subplot 2: Trade-off: Accuracy vs Latency
    colors = ['#1E6B70', '#E67E22', '#3498DB', '#95A5A6', '#9B59B6']
    scatter = ax2.scatter(latencies, accuracies, c=colors, s=250, edgecolors='black', linewidth=1.5, zorder=5)
    
    for i, txt in enumerate(clean_labels):
        ax2.annotate(txt.replace('\n', ' '), 
                     (latencies[i], accuracies[i]),
                     xytext=(10, 5), textcoords='offset points',
                     fontsize=9, fontweight='bold')

    ax2.set_xlabel('CPU Inference Latency per Document (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Efficiency Trade-off: Accuracy vs. Inference Latency', fontsize=12, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    run_ablation_study(samples_per_class=100)
