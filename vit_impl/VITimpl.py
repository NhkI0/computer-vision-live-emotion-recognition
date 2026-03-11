import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import timm

from vit_impl.train import train
from data.EmotionDataset import CLASSES, EmotionDataset

#======================

BATCH_SIZE = 16
PATH = "dataset/train"

#======================

def VITimpl():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    dataset = EmotionDataset(PATH,classes=CLASSES)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = timm.create_model('vit_small_patch16_224', pretrained=True)

    n_features = model.head.in_features
    model.head = nn.Linear(n_features, len(CLASSES))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.head.parameters(), lr=0.001, momentum=0.9)

    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device=device, name = "vit_faces.pth")

VITimpl()

