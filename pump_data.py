import os
import random
import shutil
from PIL import Image
from torchvision import transforms

# --- 1. ตั้งค่า Path โฟลเดอร์ ---
source_dir = r'C:\AI_Skin_Cancer_Project\data\train'  # โฟลเดอร์รูปปัจจุบันของคุณ
train_dir = r'C:\AI_Skin_Cancer_Project\data\balanced_train' # โฟลเดอร์ Train ที่จะปั๊มรูป
test_dir = r'C:\AI_Skin_Cancer_Project\data\test'            # โฟลเดอร์ Test ที่ห้ามปั๊มเด็ดขาด

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# ท่าพลิกแพลงสำหรับปั๊มรูป (Data Augmentation)
augment_pipeline = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
])

print("🚀 เริ่มกระบวนการแยกข้อมูล 80/20 และปั๊มรูปภาพ...")

# --- 2. สร้างโฟลเดอร์ใหม่ให้เรียบร้อย ---
for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

# --- 3. แยกข้อมูล 80/20 ---
train_counts = {}
for c in classes:
    class_path = os.path.join(source_dir, c)
    if not os.path.exists(class_path):
        continue
        
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.seed(42) # ล็อก Seed ให้ผลลัพธ์คงที่
    random.shuffle(images)
    
    # แบ่ง 80/20
    split_idx = int(len(images) * 0.8)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]
    
    # ก๊อปปี้ไฟล์ไปลงโฟลเดอร์ใหม่
    for img in train_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, c, img))
    for img in test_imgs:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_dir, c, img))
        
    train_counts[c] = len(train_imgs)
    print(f"📁 โรค {c}: แบ่ง Train = {len(train_imgs)} รูป | Test = {len(test_imgs)} รูป")

# --- 4. หาเป้าหมายจำนวนรูปที่ต้องปั๊ม (ยึดตามโรคที่เยอะที่สุด) ---
target_count = max(train_counts.values())
print(f"\n🎯 เป้าหมายคือการปั๊มรูปทุกโรคในโฟลเดอร์ Train ให้ถึง: {target_count} รูป")

# --- 5. เริ่มกระบวนการปั๊มรูป (Physical Augmentation) ---
for c in classes:
    current_count = train_counts[c]
    if current_count >= target_count:
        print(f"✅ โรค {c} มีรูปเยอะอยู่แล้ว ({current_count} รูป) ไม่ต้องปั๊มเพิ่ม")
        continue
        
    print(f"🔄 กำลังปั๊มรูปโรค {c} เพิ่มอีก {target_count - current_count} รูป...")
    
    class_train_path = os.path.join(train_dir, c)
    original_images = [f for f in os.listdir(class_train_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    generated_count = 0
    while current_count + generated_count < target_count:
        # สุ่มหยิบรูปต้นฉบับมา 1 รูป
        img_name = random.choice(original_images)
        img_path = os.path.join(class_train_path, img_name)
        
        # เปิดรูปและทำ Augmentation
        img = Image.open(img_path).convert('RGB')
        aug_img = augment_pipeline(img)
        
        # เซฟเป็นไฟล์ใหม่
        new_filename = f"aug_{generated_count}_{img_name}"
        aug_img.save(os.path.join(class_train_path, new_filename))
        
        generated_count += 1

print("\n🎉 ปั๊มรูปภาพเสร็จสมบูรณ์ 100%! ชุดข้อมูลของคุณสมดุลแบบงานวิจัยแล้วครับ")