import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# 1. ใช้โครงสร้างเดียวกับที่ใช้เทรน (ResNet50 ตามที่คุณรันใน train_model.py)
# แก้ไขฟังก์ชัน get_model ใน test_model.py ให้เป็นแบบนี้:
def get_model(num_classes=7):
    # เปลี่ยนจาก weights='DEFAULT' เป็น None เพื่อไม่ให้มันพยายามเชื่อมต่อเน็ต
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_classes))
    return model

# 2. โหลดสมอง AI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)
# แก้ไขบรรทัดโหลดโมเดลเป็นแบบนี้ครับ:
model.load_state_dict(torch.load('skin_cancer_resnet50.pth', weights_only=True))
model.eval()

# ชื่อคลาสต้องตรงกับโฟลเดอร์ใน data\train ของคุณ
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# 3. ฟังก์ชันทำนาย
def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        # คำนวณเปอร์เซ็นต์
        probabilities = F.softmax(output, dim=1)[0] * 100 
        
        print("\n📊 ผลการวิเคราะห์ความมั่นใจของ AI:")
        for i, class_name in enumerate(classes):
            print(f"- {class_name}: {probabilities[i].item():.2f}%")
            
        _, predicted = torch.max(output, 1)
        
    return classes[predicted.item()]

# 4. รันทดสอบ
try:
    print(f"ผลการทำนายคือ: {predict_image('test_image.jpg')}")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")