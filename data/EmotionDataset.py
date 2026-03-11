import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 48
CLASSES = ["negative", "neutral", "positive"]


save_augment_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])

class EmotionDataset(Dataset):
    def __init__(self, root_path, classes, transform=None):
        self.root_path = Path(root_path)
        self.classes = classes
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.samples = []
        for class_idx, class_name in enumerate(CLASSES):
            class_path = os.path.join(root_path, class_name)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                try:
                    img = None
                    for i in range(5):
                        aug_path = os.path.join(class_path, f"{img_name}_aug{i}.jpg")

                        if not os.path.exists(aug_path):
                            if img is None:
                                img = Image.open(img_path).convert("L")
                            img_aug = save_augment_transform(img)
                            img_aug = img_aug.resize((48, 48))
                            img_aug.save(aug_path)

                        self.samples.append((aug_path, self.class_to_idx[class_name]))

                except Exception as e:
                    print("Error loading image:", img_path)
                    print(e)

        print(f"Dataset charge: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label