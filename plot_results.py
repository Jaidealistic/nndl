import json
import matplotlib.pyplot as plt
import seaborn as sns

with open('src/model/training_results.json') as f:
    data = json.load(f)

epochs = [e['epoch'] for e in data['epochs']]
train_loss = [e['train_loss'] for e in data['epochs']]
val_loss = [e['val_loss'] for e in data['epochs']]
train_acc = [e['train_acc'] for e in data['epochs']]
val_acc = [e['val_acc'] for e in data['epochs']]

# Set up the aesthetic
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss Plot
ax1.plot(epochs, train_loss, color='#1E6B70', linestyle='-', label='Training Loss', linewidth=2.5, marker='o')
ax1.plot(epochs, val_loss, color='#7B1438', linestyle='--', label='Validation Loss', linewidth=2.5, marker='s')
ax1.set_title('Training & Validation Loss (GROGU-KYC)', fontsize=14, pad=15)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=11)
ax1.set_xticks(epochs)
ax1.tick_params(axis='both', labelsize=11)

# Accuracy Plot
ax2.plot(epochs, train_acc, color='#1E6B70', linestyle='-', label='Training Accuracy', linewidth=2.5, marker='o')
ax2.plot(epochs, val_acc, color='#7B1438', linestyle='--', label='Validation Accuracy', linewidth=2.5, marker='s')
ax2.set_title('Training & Validation Accuracy', fontsize=14, pad=15)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.legend(fontsize=11)
ax2.set_xticks(epochs)
ax2.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig('results-graph.png', dpi=300, bbox_inches='tight')
print("Successfully generated results-graph.png")
