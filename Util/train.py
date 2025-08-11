import os
import torch.optim as optim
# from flatbuffers.flexbuffers import F
import torch.nn.functional as F
# from prompt_toolkit import output
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import TopKMultilabelAccuracy
from model.CNN_classifier import *

def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor, reduction="mean"):
    # logits: (N, C), target_probs: (N, C) with rows summing to 1
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss  # shape (N,)

def compute_loss(output,labels,mixup,is_multilabel):
    if is_multilabel:
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output, labels)
    else:
        if mixup:
            log_probs = F.log_softmax(output, dim=1)
            # Ensure each row sums to 1:
            labels = labels / labels.sum(dim=1, keepdim=True).clamp(1e-12)
            loss = -(labels * log_probs).sum(dim=1).mean()  # soft cross-entropy
            # loss = soft_cross_entropy()
        else:
            if labels.ndim == 2:
                labels = labels.argmax(dim=1)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, labels)
    return loss


def main_train_loop(train_loader, val_loader,mixup, num_epochs, lr, batch_size, dropout, device, best_model_dir,classes_num,is_multilabel):
    writer = SummaryWriter(log_dir=f'../run_log/epochs={num_epochs}_lr={lr:1.5f}_batch_size={batch_size}_dropout={dropout:1.2f}')

    # Check if a model already exists. If it does, use its best validation accuracy as a benchmark to find a better model.
    if os.path.exists(f'../' + best_model_dir + '/model_full.pth'):
        checkpoint = torch.load('../' + best_model_dir +'/model_full.pth', map_location='cpu')
        best_val_acc = checkpoint['best_val_acc']
    else:
        best_val_acc = 0
    train_losses = []
    val_accuracies = []
    log_interval = 16

    # Build model
    model = CNN_classifier(1, classes_num, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = StepLR(optimizer, step_size=600, gamma=0.1)
    metric = TopKMultilabelAccuracy(k=2, criteria="exact_match").to(device) if is_multilabel else None

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
            # labels = labels.to(output.dtype)
            loss = compute_loss(output, labels, mixup, is_multilabel)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batch += 1

            # if (batch % log_interval == 0) & (batch > 0):
            #     print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
            #           f'lr: {lr:02.6f} | loss: {running_loss/log_interval:5.3f}')

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
                # labels = labels.to(output.device).long()  # indices



                # To Compute the accuracy
                if is_multilabel:
                    # For multi-label classification, use BCEWithLogitsLoss
                    # probs = torch.sigmoid(output)
                    # topk_preds = torch.zeros_like(probs, dtype=torch.int)
                    # topk_indices = torch.topk(probs, k=2, dim=1).indices
                    # topk_preds.scatter_(1, topk_indices, 1)  # mark top-k positions as 1
                    metric.update(output, labels)
                    total += labels.size(0)
                else:
                    # running_val_loss += compute_loss(output, labels,mixup,is_multilabel)
                    if labels.ndim == 2 and labels.size(1) == output.size(1):
                        targets = labels.argmax(dim=1)  # (N,)
                    else:  # already indices
                        targets = labels.long().view(-1)

                    preds = output.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += labels.size(0)


                # _, predicted = torch.max(output.data, 1)
                # if mixup:
                #     _, target = torch.max(labels.data, 1)
                # else:
                #     target = labels.data

                #To compute the loss
                loss = compute_loss(output, labels,mixup,is_multilabel)
                running_val_loss += loss.item()
                # total += labels.size(0)
                # correct += torch.sum(predicted == target).item()

                # else:
                #     # For multi-class classification, use CrossEntropyLoss
                #     _, predicted = torch.topk(output, 1, dim=1, largest=True)
                #     # labels = torch.argmax(labels, dim=1)
                #     correct += (predicted == torch.argmax(labels.unsqueeze(1), dim=1)).sum().item()
                #     total += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100*metric.compute().item() if is_multilabel else 100 * correct / total
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            # best_model = model
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

        # scheduler.step()

        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | valid loss {val_loss:.3f} | valid accuracy {val_accuracy:.3f}')
        print('-' * 89)

        # TensorBoard
        writer.add_scalar("Loss/valid", running_val_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/train", running_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy/valid", val_accuracy, epoch)
        writer.close()

