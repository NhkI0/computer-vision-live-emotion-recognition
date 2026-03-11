import torch
from tqdm import tqdm

import os
import torch
import torch.multiprocessing

#======================

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device,name):
    if device == torch.device("cpu"):
        torch.set_num_threads(1)
        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    model.to(device).float()
    best_acc = 0.0
    for epoch in range(num_epochs):

        model.train()


        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
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
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            # validatioin
            for inputs, labels in val_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device)

                outputs = model(inputs)
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
