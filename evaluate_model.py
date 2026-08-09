import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 1. เตรียม Data Pipeline (ต้องใช้การ Normalize แบบเดิม)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ระบุ Path ให้ตรงกับที่คุณเก็บข้อมูลไว้ (แก้ไขให้ตรงของจริง)
test_dataset = datasets.ImageFolder(root=r'C:\AI_Skin_Cancer_Project\data\train', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. สร้างโครงสร้างสมอง (ต้องเหมือนกับตอนที่เราเทรนเป๊ะๆ)
model = models.resnet50(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),       
    nn.ReLU(),
    nn.Dropout(0.5),           
    nn.Linear(512, 7)          
)

# 3. โหลด "ร่างทอง" ที่เราเพิ่งเทรนเสร็จ
model.load_state_dict(torch.load('skin_cancer_bestv2_resnet.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval() # ปิดโหมดเทรน เพื่อเข้าโหมดทำข้อสอบ

# 4. เริ่มทำข้อสอบและเก็บคะแนน
all_preds = []
all_labels = []

print("กำลังให้ AI ทำข้อสอบ กรุณารอสักครู่...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 5. สร้างรายงานผล (Classification Report)
classes = test_dataset.classes
print("\n" + "="*50)
print("📊 Classification Report (ตารางสรุปผลความแม่นยำ)")
print("="*50)
print(classification_report(all_labels, all_preds, target_names=classes))

# 6. วาดกราฟ Confusion Matrix ด้วย Seaborn
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix: Skin Cancer Detection (ResNet50)')
plt.ylabel('เฉลยที่ถูกต้อง (True Label)')
plt.xlabel('สิ่งที่ AI ทาย (Predicted Label)')
plt.show()