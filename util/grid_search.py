import argparse
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from util.augmented_mfcc_dataset import load_audio_paths_and_labels, create_split_masks_stratified, \
    AugmentedMFCCDataset, create_split_masks
from util.data_import import download_data
from util.test import evaluate_model_on_test
from util.train import main_train_loop

BEST_MODEL_DIR = 'best_model'
DATA_DIR = "../data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_dir', type=str, default=BEST_MODEL_DIR)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mixup', type=bool, default=False)
    parser.add_argument('--augment', type=bool, default=False)
    parser.add_argument('--is_multilabel', type=bool, default=False)
    parser.add_argument('--stratify', type=bool, default=True)
    parser.add_argument('--test_augment', type=bool, default=False)
    parser.add_argument('--should_download_data', type=bool, default=False)
    parser.add_argument('--search_attempts', type=int, default=10)
    parser.add_argument('--drone_fine_tune', type=bool, default=False)
    return parser.parse_args()


def grid_search(best_model_dir, data_dir, search_attempts,drone_fine_tune, mixup, augment, is_multilabel,test_augment,
                should_download_data,stratify, seed=0):
    # Set fixed seed
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)  # For CPU ops

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if should_download_data:
        download_data(
            data_dir)  # Download original .wav files to data dir from "janboubiabderrahim/vehicle-sounds-dataset".

    classes_num = len(os.listdir(data_dir))

    if drone_fine_tune:
        is_multilabel = False
        mixup = False
        if is_multilabel:
            print("Drone Fine-Tuning conflicts with multilabel- is it a binary classifier")



    # Get files paths and labels, split data
    audio_paths, labels, label_map = load_audio_paths_and_labels(data_dir)
    if stratify:
        train_idx, val_idx, test_idx = create_split_masks_stratified(labels,
                                                                     train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                                                                     seed=seed)
    else:
        train_idx, val_idx, test_idx = create_split_masks(data_dir, train_ratio=0.7, val_ratio=0.2,
                                                          test_ratio=0.1, seed=seed)

    # Training dataset
    training_paths = [audio_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    train_audio_augmentation_params = {
        'pitch_shift': [-3, 3],
        'add_noise': [0.001, 0.1],
        'shift_max': 0.2,  # seconds
        'volume_gain_dB': [-4, 4],
        'reverb': False #For reverb, pass rir files dir to dataset rir_dir.
    }

    mfcc_augmentation_params = {
        'add_noise': [0.001, 0.01],
        'dropout': 0.1
    }

    train_dataset = AugmentedMFCCDataset(training_paths, train_labels, label_map, mixup=mixup, audio_augment=augment,
                                         audio_params=train_audio_augmentation_params, mfcc_augment=augment,
                                         mfcc_params=mfcc_augmentation_params, training=True, classes_num=classes_num,
                                         is_multilabel=is_multilabel,drone_fine_tune=drone_fine_tune)
    train_dataset.preprocess(seed)

    # Validation dataset
    test_audio_augmentation_params = {
        # 'pitch_shift': [-3,3],
        'add_noise': [0.001, 0.1],
        'shift_max': 0.2,  # seconds
        'volume_gain_dB': [-4, 4],
        # 'reverb': True
    }

    validating_paths = [audio_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_dataset = AugmentedMFCCDataset(validating_paths, val_labels, label_map, training=False, mixup=mixup,
                                       classes_num=classes_num, is_multilabel=is_multilabel, audio_augment=test_augment,
                                       audio_params=test_audio_augmentation_params,drone_fine_tune=drone_fine_tune)
    val_dataset.preprocess(seed)

    # Randomized grid search loop
    for trial in range(search_attempts):
        num_epochs = 150
        lr = 10 ** random.uniform(-5, -3)
        batch_size = 2 ** random.randint(5, 8)
        dropout = random.uniform(0, 0.2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        main_train_loop(train_loader, val_loader,drone_fine_tune, mixup, num_epochs, lr, batch_size, dropout, device,
                        best_model_dir, classes_num, is_multilabel=is_multilabel)
        print("trial number", trial)

    # =====Testing=====
    testing_paths = [audio_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    testset = AugmentedMFCCDataset(testing_paths, test_labels, label_map, training=False, classes_num=classes_num,
                                   is_multilabel=is_multilabel, audio_augment=test_augment,
                                   audio_params=test_audio_augmentation_params,drone_fine_tune=drone_fine_tune)
    testset.preprocess(seed)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)

    evaluate_model_on_test(test_loader, '../' + best_model_dir, device, k=2, criteria="exact_match",
                           is_multilabel=is_multilabel,drone_fine_tuning=drone_fine_tune)


if __name__ == '__main__':
    args = parse_args()
    script_dir = Path(__file__).parent
    parent_of_code_folder = script_dir.parent
    best_model_dir = parent_of_code_folder / args.best_model_dir
    Path(best_model_dir).mkdir(parents=True, exist_ok=True)
    grid_search(best_model_dir=args.best_model_dir, data_dir=args.data_dir, mixup=args.mixup, augment=args.augment,
                is_multilabel=args.is_multilabel,
                test_augment=args.test_augment, should_download_data=args.should_download_data, stratify=args.stratify,
                seed=args.seed, search_attempts=args.search_attempts,drone_fine_tune=args.drone_fine_tune)
