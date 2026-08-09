import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import time

# 1. การปรับปรุงข้อมูล (ปรับขนาด 28x28 และ Normalization 0-1 ตามเปเปอร์)
data_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), # แปลงเป็น Tensor ซึ่งจะปรับค่าสีให้อยู่ในช่วง 0-1 อัตโนมัติ
])

# ระบุพาร์ทโฟลเดอร์ train
full_dataset = datasets.ImageFolder(root=r'C:\AI_Skin_Cancer_Project\data\train', transform=data_transforms)

# 2. ทำ Oversampling เกลี่ยข้อมูลแก้ปัญหา Class Imbalance
targets = [s[1] for s in full_dataset.samples]
class_count = np.bincount(targets)
class_weights = 1. / class_count
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(full_dataset, batch_size=32, sampler=sampler)

# 3. สถาปัตยกรรม Custom CNN
class DermascanCNN(nn.Module):
    def __init__(self):
        super(DermascanCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 7)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DermascanCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# เพิ่มระบบ Scheduler (ลด Learning Rate ทันทีถ้า Loss เริ่มนิ่ง)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 4. ลูปการฝึกสอน (Training Loop) พร้อม Early Stopping
start_time = time.time()
best_loss = float('inf')
patience = 15 # เพิ่มความอดทนเป็น 15 รอบ
counter = 0

print("🚀 เริ่มเทรนโมเดล Custom CNN (ครบเครื่อง: Oversampling + Scheduler + Accuracy)...")
for epoch in range(300): # ตั้งไว้ 300 รอบ เผื่อให้มันค่อยๆ เรียนรู้
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # คำนวณความแม่นยำ
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # สรุปผลราย Epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    # อัปเดต Scheduler
    scheduler.step(avg_loss)
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | LR: {current_lr:.6f}")

    # ระบบ Early Stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        counter = 0
        torch.save(model.state_dict(), 'skin_cancer_best_custom.pth')
    else:
        counter += 1
        print(f"⚠️ Loss ไม่ลดลงเป็นรอบที่ {counter}/{patience}")
        if counter >= patience:
            print("🛑 ทำงาน Early Stopping! หยุดการเทรนเพื่อป้องกัน Overfitting")
            break

# สรุปเวลา
elapsed_time = time.time() - start_time
print(f"✅ เทรนเสร็จสิ้น! บันทึกไฟล์ 'skin_cancer_best_custom.pth' เรียบร้อย ใช้เวลาไป: {elapsed_time/60:.2f} นาที")