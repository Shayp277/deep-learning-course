import os.path
from torch.utils.data import DataLoader
from model.CNN_classifier import nn
from Util.AugmentedMFCCDataset import *
from torcheval.metrics import TopKMultilabelAccuracy


def evaluate_model_on_test(test_loader, model_path, device, k=2, criteria="exact_match", is_multilabel=False):
    """
    Evaluate the model on the given data loader.

    Args:
        model_path: The path where the PyTorch model to evaluate is saved.
        test_loader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.
        k: Number of top predictions to consider.
        criteria: Criteria for matching top-k predictions ("exact_match", "hamming", etc.).
        is_multilabel: Flag indicating if the task is multi-class (False) or multi-label (True).

    Returns:
        accuracy: The computed accuracy.
    """
    # Load model
    checkpoint = torch.load(os.path.join(model_path, 'model_full.pth'), map_location=device)
    model = checkpoint['model']
    metric = TopKMultilabelAccuracy(k=k, criteria=criteria).to(device) if is_multilabel else None
    model.eval()
    criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss() #criterion: Loss function used during training.
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            #calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            if is_multilabel:
                # For multi-label classification, use BCEWithLogitsLoss
                metric.update(outputs, labels)
                total += labels.size(0)

            else:
                # For multi-class classification, use CrossEntropyLoss
                _, predicted = torch.topk(outputs, 1, dim=1, largest=True, sorted=True)
                labels = torch.argmax(labels, dim=1)
                correct += (predicted == labels.unsqueeze(1)).sum().item()
                total += labels.size(0)

            # _, predicted = torch.max(outputs.data, 1)
            # if test_loader.dataset.mixup:
            #     _, target = torch.max(labels.data, 1)
            # else:
            #     target = labels.data

            # correct += torch.sum(predicted == target).item()
            # total += labels.size(0)

    # accuracy = 100 * correct / total
    # avg_loss = test_loss / total

    avg_loss = test_loss / len(test_loader)
    accuracy = metric.compute().item() if is_multilabel else 100 * correct / total
    title0 = "multilabel" if is_multilabel else "singlelabel"
    print(title0 + f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    root = "../data"
    audio_paths, labels, label_map = load_audio_paths_and_labels(root)
    train_idx, val_idx, test_idx = create_split_masks(root, train_ratio=0.7, val_ratio=0.2,
                                                         test_ratio=0.1, seed=0)
    testing_paths = [audio_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_loader = DataLoader(AugmentedMFCCDataset(testing_paths, test_labels, label_map, training=False,mixup=False), batch_size=1, shuffle=False)


    # test model
    best_model_dir = 'shay_best_model'
    evaluate_model_on_test(test_loader, '../' + best_model_dir, device)

if __name__ == '__main__':
    main()