import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import timm

from vit_impl.train import train
from data.EmotionDataset import CLASSES, EmotionDataset,save_augment_transform

#======================

BATCH_SIZE = 16
PATH = "dataset/train"

#======================

class VITimpl():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        print("loading the data...")
        dataset = EmotionDataset(PATH,classes=CLASSES, transform=save_augment_transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        
        print("initializing the model...")
        self.model = timm.create_model('vit_small_patch16_224', pretrained=True)

        n_features = model.head.in_features
        self.model.head = nn.Linear(n_features, len(CLASSES))
        self.model = model.to(device)
        print("The model's ready")



    def train(self, num_epochs = 5, name = "vit_faces.pth"):
        print("starting the training")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.head.parameters(), lr=0.001, momentum=0.9)
        if self.device == torch.device("cpu"):
            torch.set_num_threads(1)
            torch.multiprocessing.set_sharing_strategy('file_system')
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
        self.model.to(device).float()
        best_acc = 0.0
        for epoch in range(num_epochs):

            self.model.train()


            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # acc & loss for this epoch
            train_loss = running_loss / len(train_loader.dataset)
            train_acc = running_corrects.float() / len(train_loader.dataset)

            # eval on validation
            self.model.eval()
            running_loss = 0.0
            running_corrects = 0

            with torch.no_grad():
                # validation
                for inputs, labels in val_loader:
                    inputs = inputs.to(device).float()
                    labels = labels.to(device)

                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # value needed for metrics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            # acc & loss for this epoch
            val_loss = running_loss / len(val_loader.dataset)
            val_acc = running_corrects.float() / len(val_loader.dataset)

            # Print the results for the current epoch

            print(f'Epoch [{epoch+1}/{num_epochs}], train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"models/{name}")
                print(f"--> Meilleur modèle sauvegardé ({name}) avec acc: {val_acc:.4f}")

        print(f"Entraînement terminé. Meilleure précision : {best_acc:.4f}")

        def load_model(self,path = "models/vit_faces.pth"):
            """
                load an existing model from the weights
            """
            self.model = timm.create_model("vit_small_patch16_224", num_classes = len(CLASSES))
            self.model.load_state_dict(torch.load(path,map_location = self.device)))

        def guess(self,img,transform):
            """
                guess an emotion from an image
                input
                    img : image
                    transform : trasformation fuction
                output
                    emotion type : string
            """
            img = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self.model(img)
                useless, predicted = torch.max(output,1)
            return CLASSES[predicted.item()]

vit = VITimpl()
vit.train()

