# computer-vision-live-emotion-recognition

### EmotionDataset

```python
from data.EmotionDataset import EmotionDataset, CLASSES
from torchvision import transforms

# Définir les transformations pour l'entraînement
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Créer le dataset
dataset = EmotionDataset(
    root_path="dataset/train",
    classes=CLASSES,
    transform=train_transform
)

# Utiliser avec DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Le dataset génère automatiquement 5 versions augmentées de chaque image (flip horizontal, rotation, variation de couleur) et les sauvegarde sur le disque pour un chargement plus rapide lors des utilisations suivantes.

### Nettoyage des images augmentées

Pour supprimer toutes les images augmentées générées :

```python
from data.data_loader import remove_augmented_images

# Supprimer les images augmentées du chemin par défaut
remove_augmented_images()

# Ou spécifier un chemin personnalisé
remove_augmented_images(train_path="chemin/vers/train")
```