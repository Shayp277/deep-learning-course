import argparse
import os.path
import random

import torch
from torch.utils.data import DataLoader
from torcheval.metrics import TopKMultilabelAccuracy

from train import compute_loss
from util.augmented_mfcc_dataset import load_audio_paths_and_labels, create_split_masks, create_split_masks_stratified, \
    AugmentedMFCCDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--is_multilabel', type=bool, default=False)
    parser.add_argument('--stratify', type=bool, default=True)
    return parser.parse_args()


def evaluate_model_on_test(test_loader, model_path, device, k=1, criteria="exact_match", is_multilabel=False,
                           mixup=None):
    """
    Evaluate the model on the given data loader.

    Args:
        model_path: The path where the PyTorch model to evaluate is saved.
        test_loader: DataLoader for the evaluation dataset.
        device: Device to run the evaluation on.
        k: Number of top predictions to consider.
        criteria: Criteria for matching top-k predictions ("exact_match", "hamming", etc.).
        is_multilabel: Flag indicating if the task is multi-class (False) or multi-label (True).
        mixup: Flag indicating if the test data was mixed.
    Returns:
        accuracy: The computed accuracy.
    """
    # Load model
    checkpoint = torch.load(os.path.join(model_path, 'model_full.pth'), map_location=device)
    model = checkpoint['model']
    metric = TopKMultilabelAccuracy(k=k, criteria=criteria).to(device) if is_multilabel else None
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    # Test loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # calculate loss
            loss = compute_loss(outputs, labels, mixup, is_multilabel)
            test_loss += loss.item()

            if is_multilabel:
                metric.update(outputs, labels)
                total += labels.size(0)

            else:
                # Convert labels from hot-ones to indices
                if labels.ndim == 2 and labels.size(1) == outputs.size(1):
                    targets = labels.argmax(dim=1)  # (N,)
                else:  # already indices
                    targets = labels

                prediction = outputs.argmax(dim=1)
                correct += (prediction == targets).sum().item()
                total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * metric.compute().item() if is_multilabel else 100 * correct / total
    title0 = "multilabel" if is_multilabel else "single label"

    print(title0 + f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss


def main(model_dir,data_dir,is_multilabel,test_augment,classes_num,seed,stratify=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    audio_paths, labels, label_map = load_audio_paths_and_labels(data_dir)
    if stratify:
        train_idx, val_idx, test_idx = create_split_masks_stratified(labels,
                                                                     train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                                                                     seed=seed)
    else:
        train_idx, val_idx, test_idx = create_split_masks(data_dir, train_ratio=0.7, val_ratio=0.2,
                                                          test_ratio=0.1, seed=seed)
    test_audio_augmentation_params = {
        # 'pitch_shift': [-3,3],
        'add_noise': [0.001, 0.1],
        'shift_max': 0.2,  # seconds
        'volume_gain_dB': [-4, 4],
        # 'reverb': True
    }

    testing_paths = [audio_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    testset = AugmentedMFCCDataset(testing_paths, test_labels, label_map, training=False, classes_num=classes_num,
                                   is_multilabel=is_multilabel, audio_augment=test_augment,
                                   audio_params=test_audio_augmentation_params)
    test_loader = DataLoader(testset,batch_size=1, shuffle=False)

    # test model
    evaluate_model_on_test(test_loader, '../' + model_dir, device)


if __name__ == '__main__':
    args = parse_args()
    main(args.model_dir, args.data_dir, args.is_multilabel, args.test_augment, args.classes_num, args.seed)
