import os
import glob

from data.EmotionDataset import EmotionDataset

IMG_SIZE = 48
TRAIN_PATH = "../dataset/train"
CLASSES = ["negative", "neutral", "positive"]


def remove_augmented_images(train_path=TRAIN_PATH):
    removed_count = 0
    for class_name in CLASSES:
        class_path = os.path.join(train_path, class_name)
        aug_files = glob.glob(os.path.join(class_path, "*_aug*.jpg"))
        for f in aug_files:
            os.remove(f)
            removed_count += 1
    print(f"{removed_count} images supprimées")
    return removed_count

if __name__ == "__main__":
    dataset = EmotionDataset(root_path=TRAIN_PATH, classes=CLASSES)