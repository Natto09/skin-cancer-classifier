import os
import random
import shutil

# --- ตั้งค่าโฟลเดอร์ ---
source_test_dir = r'C:\AI_Skin_Cancer_Project\data\test'
balanced_test_dir = r'C:\AI_Skin_Cancer_Project\data\balanced_test'

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# 1. หาว่าโรคไหนมีจำนวนรูปน้อยที่สุด
counts = {c: len(os.listdir(os.path.join(source_test_dir, c))) for c in classes}
min_samples = min(counts.values()) 
print(f"📊 โรคที่น้อยที่สุดมี {min_samples} รูป เราจะปรับทุกโรคให้เท่ากันที่จำนวนนี้ครับ")

# 2. สุ่มคัดเลือกรูปให้เท่ากันเป๊ะ
for c in classes:
    os.makedirs(os.path.join(balanced_test_dir, c), exist_ok=True)
    all_files = os.listdir(os.path.join(source_test_dir, c))
    
    # สุ่มเลือกมาให้เท่ากับ min_samples
    selected_files = random.sample(all_files, min_samples)
    
    for f in selected_files:
        shutil.copy(os.path.join(source_test_dir, c, f), os.path.join(balanced_test_dir, c, f))
    print(f"✅ จัดชุด Test โรค {c} เรียบร้อย: {min_samples} รูป")