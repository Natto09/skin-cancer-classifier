import matplotlib.pyplot as plt
import re

# 1. วางข้อความจาก Terminal ของคุณลงไปตรงนี้ 
# (ผมจำลองตัวเลขบางส่วนจากรูปของคุณให้ดูเป็นตัวอย่าง คุณสามารถเอาของจริงทั้งหมดมาแปะทับได้เลยครับ)
terminal_output = """
Epoch 01 | Train Loss: 0.5438 Acc: 80.83% | Val Loss: 0.5028 Acc: 82.57%
Epoch 02 | Train Loss: 0.1547 Acc: 94.91% | Val Loss: 0.4676 Acc: 84.29%
Epoch 03 | Train Loss: 0.0800 Acc: 97.33% | Val Loss: 0.5103 Acc: 84.66%
Epoch 04 | Train Loss: 0.0526 Acc: 98.41% | Val Loss: 0.5655 Acc: 85.15%
Epoch 05 | Train Loss: 0.0478 Acc: 98.65% | Val Loss: 0.5996 Acc: 85.25%
Epoch 06 | Train Loss: 0.0192 Acc: 99.52% | Val Loss: 0.5288 Acc: 86.70%
Epoch 07 | Train Loss: 0.0101 Acc: 99.79% | Val Loss: 0.5204 Acc: 86.38%
"""

# 2. ให้ AI ดึงข้อมูลอัตโนมัติ
epochs, losses, accuracies = [], [], []
for line in terminal_output.strip().split('\n'):
    if "Epoch" in line and "Loss:" in line:
        epochs.append(int(re.search(r'Epoch (\d+)', line).group(1)))
        losses.append(float(re.search(r'Loss: ([\d.]+)', line).group(1)))
        accuracies.append(float(re.search(r'Accuracy: ([\d.]+)', line).group(1)))

# 3. วาดกราฟแบบงานวิจัย 
plt.figure(figsize=(14, 5))

# กราฟซ้าย: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracies, color='#1f77b4', marker='o', label='Training Accuracy', linewidth=2)
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# กราฟขวา: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, losses, color='#ff7f0e', marker='o', label='Training Loss', linewidth=2)
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()