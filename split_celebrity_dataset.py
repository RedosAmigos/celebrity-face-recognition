import os
import shutil
import random

def split_dataset(base_dir="Celebrity Faces Dataset", output_dir="celebrity_dataset_split", 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    train_dir = os.path.join(output_dir, "train")
    val_dir   = os.path.join(output_dir, "val")
    test_dir  = os.path.join(output_dir, "test")

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for celeb_name in os.listdir(base_dir):
        celeb_folder = os.path.join(base_dir, celeb_name)
        if not os.path.isdir(celeb_folder):
            continue

        images = [f for f in os.listdir(celeb_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 3:
            continue  # skip datasets too small to split

        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        os.makedirs(os.path.join(train_dir, celeb_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, celeb_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, celeb_name), exist_ok=True)

        for img in train_images:
            shutil.copy2(os.path.join(celeb_folder, img), os.path.join(train_dir, celeb_name, img))
        for img in val_images:
            shutil.copy2(os.path.join(celeb_folder, img), os.path.join(val_dir, celeb_name, img))
        for img in test_images:
            shutil.copy2(os.path.join(celeb_folder, img), os.path.join(test_dir, celeb_name, img))

        print(f"[INFO] {celeb_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    print("\n✅ Dataset split completed!")

if __name__ == "__main__":
    split_dataset()