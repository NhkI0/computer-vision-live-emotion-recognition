# Computer Vision — Live Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/github/license/NhkI0/computer-vision-live-emotion-recognition)
![Last Commit](https://img.shields.io/github/last-commit/NhkI0/computer-vision-live-emotion-recognition)

Système de reconnaissance d'émotions faciales en temps réel basé sur un CNN entraîné avec PyTorch. Le modèle classifie les expressions faciales en 3 catégories : **Negative**, **Neutre**, **Positive**.

## Table des matières

- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Détection en temps réel (webcam)](#détection-en-temps-réel-webcam)
  - [Entraînement du modèle](#entraînement-du-modèle)
  - [Prédiction sur une image](#prédiction-sur-une-image)
- [Architecture du modèle CNN](#architecture-du-modèle-cnn)
- [Dataset et augmentation](#dataset-et-augmentation)
- [Configuration](#configuration)

## Structure du projet

```
├── data/
│   ├── EmotionDataset.py          # Dataset PyTorch avec augmentation automatique
│   └── data_loader.py             # Utilitaires de chargement et nettoyage
├── models/
│   ├── cnn/
│   │   ├── CNN.py                 # Architecture EmotionCNN
│   │   ├── training.py            # Pipeline d'entraînement
│   │   ├── predict.py             # Module d'inférence
│   │   └── best_model.pth         # Poids du modèle entraîné
│   └── haarcascade_frontalface_default.xml  # Détection de visage Haar Cascade
├── vit_impl/
│   └── VITimpl.py                 # Vision Transformer (expérimental)
├── dataset/
│   ├── train/{negative,neutral,positive}/
│   └── test/{negative,neutral,positive}/
├── webcam.py                      # Détection en temps réel via webcam
├── requirements.txt               # Dépendances
└── README.md
```

## Installation

```bash
git clone https://github.com/NhkI0/computer-vision-live-emotion-recognition.git
cd computer-vision-live-emotion-recognition
pip install -r requirements.txt
```

### Dépendances

| Package | Usage |
|---------|-------|
| `torch` / `torchvision` | Modèle CNN, entraînement, inférence |
| `opencv-python` | Capture webcam, détection de visage |
| `numpy` | Manipulation de tableaux |
| `matplotlib` | Visualisation des métriques |
| `pillow` | Chargement et transformation d'images |
| `tqdm` | Barres de progression pendant l'entraînement |

> Le projet supporte automatiquement **MPS** (Apple Silicon), **CUDA** (NVIDIA) et **CPU**.

## Utilisation

### Détection en temps réel (webcam)

```bash
python webcam.py
```

- Les visages détectés sont encadrés avec un code couleur :
  - 🟢 **Vert** → Positive
  - 🔴 **Rouge** → Negative
  - 🟠 **Orange** → Neutre
- L'émotion et le pourcentage de confiance sont affichés au-dessus de chaque visage
- Appuyer sur **ESC** pour quitter

### Entraînement du modèle

```bash
# Entraînement complet
python models/cnn/training.py

# Entraînement sur un sous-ensemble (10% du dataset)
python models/cnn/training.py --sample 0.1
```

Le meilleur modèle est sauvegardé automatiquement dans `models/cnn/best_model.pth`.

**Paramètres d'entraînement :**

| Paramètre | Valeur |
|-----------|--------|
| Epochs | 50 (early stopping, patience=10) |
| Batch size | 64 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| Loss | CrossEntropyLoss avec pondération des classes |
| Split train/val | 80/20 (seed=42) |

### Prédiction sur une image

```python
from models.cnn.predict import predict_emotion
import cv2

image = cv2.imread("face.jpg")
emotion, confidence = predict_emotion(image)
print(f"{emotion}: {confidence:.2%}")
```

`predict_emotion` retourne un tuple `(str, float)` — le nom de l'émotion et le score de confiance entre 0 et 1.

## Architecture du modèle CNN

Le modèle `EmotionCNN` prend en entrée des images en niveaux de gris **48×48 pixels** et produit une classification parmi 3 émotions.

```
Entrée (1×48×48)
    │
    ├─ Conv2d(1→64)   + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ├─ Conv2d(64→128)  + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ├─ Conv2d(128→256) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ├─ Conv2d(256→512) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    ├─ Conv2d(512→512) + BatchNorm + ReLU + MaxPool + Dropout(0.3)
    │
    ├─ Flatten
    ├─ Linear(512→512) + BatchNorm1d + ReLU + Dropout(0.3)
    ├─ Linear(512→512) + BatchNorm1d + ReLU + Dropout(0.3)
    └─ Linear(512→3)
        │
    Sortie (3 classes)
```

## Dataset et augmentation

### EmotionDataset

```python
from data.EmotionDataset import EmotionDataset, CLASSES
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = EmotionDataset(
    root_path="dataset/train",
    classes=CLASSES,
    transform=train_transform
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Le dataset génère automatiquement **5 versions augmentées** de chaque image et les sauvegarde sur le disque :

| Augmentation | Paramètres |
|-------------|------------|
| Flip horizontal | Aléatoire |
| Rotation | ±15° |
| Color jitter | brightness=0.2, contrast=0.2, saturation=0.2 |

### Nettoyage des images augmentées

```python
from data.data_loader import remove_augmented_images

remove_augmented_images()

# Ou avec un chemin personnalisé
remove_augmented_images(train_path="chemin/vers/train")
```

## Configuration

Paramètres ajustables dans `webcam.py` :

| Variable | Défaut | Description |
|----------|--------|-------------|
| `CLAHE` | `False` | Amélioration du contraste (CLAHE) |
| `SMOOTHING` | `0.3` | Facteur de lissage du bounding box entre frames |
| `minNeighbors` | `7` | Sensibilité de détection Haar Cascade |
| `minSize` | `(80, 80)` | Taille minimale de visage détecté |
