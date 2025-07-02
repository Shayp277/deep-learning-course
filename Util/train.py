import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from Util.AugmentedMFCCDataset import *
from model.CNN_classifier import*
from torch.utils.tensorboard import SummaryWriter


def main_train_loop(train_loader, val_loader,mixup, num_epochs, lr, batch_size, dropout, device, best_model_dir):
    writer = SummaryWriter(log_dir=f'../aug_run/epochs={num_epochs}_lr={lr:1.5f}_batch_size={batch_size}_dropout={dropout:1.2f}')

    # Check if a model already exists. If it does, use its best validation accuracy as a benchmark to find a better model.
    if os.path.exists(f'../' + best_model_dir + '/model_full.pth'):
        checkpoint = torch.load('../' + best_model_dir +'/model_full.pth', map_location='cpu')
        best_val_acc = checkpoint['best_val_acc']
    else:
        best_val_acc = 0
    best_model = None
    train_losses = []
    val_accuracies = []
    log_interval = 16

    # Build model
    model = CNN_classifier(1, 8, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if mixup:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # train loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += 1

            if (batch % log_interval == 0) & (batch > 0):
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr: {lr:02.6f} | loss: {running_loss/log_interval:5.3f}')

        train_losses.append(running_loss / len(train_loader))

        # Validation accuracy
        running_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                output = model(inputs)
                _, predicted = torch.max(output.data, 1)
                _, target = torch.max(labels, 1)
                loss = criterion(output, labels)
                running_val_loss += loss.item()
                total += labels.size(0)
                correct += torch.sum(predicted == target).item()

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model = model
            torch.save(
                {'model': model,
                'best_val_acc': best_val_acc,
                 'model_params': {
                     'lr': lr,
                     'dropout': dropout,
                     'batch_size': batch_size,
                     'num_epochs': num_epochs
                 }
            },'../' + best_model_dir + '/model_full.pth')

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:.3f} | valid accuracy {val_accuracy:.3f}')
        print('-' * 89)

        # TensorBoard
        writer.add_scalar("Loss/valid", running_val_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        writer.close()

def load_audio_paths_and_labels(root_dir):
    class_names = (os.listdir(root_dir))  # Sorted = consistent label order
    label_map = {class_name: i for i, class_name in enumerate(class_names)}

    audio_paths = []
    labels = []

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(".wav"):
                filepath = os.path.join(class_dir, filename)
                audio_paths.append(filepath)
                labels.append(label_map[class_name])

    return audio_paths, labels, label_map