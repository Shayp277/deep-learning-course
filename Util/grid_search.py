import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from Util.test import evaluate_model_on_test
from Util.train import main_train_loop
from torch.utils.data import DataLoader
from Util.AugmentedMFCCDataset import *

random.seed(0)
torch.cuda.manual_seed_all(0)

def grid_search():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_dir = 'shay_best_model'  # choose dir to save the current best model
    mixup = False
    augment = True
    # download_data()  # if you need to create new dataset you need to download wav file
    root = "../data"
    audio_paths, labels, label_map = load_audio_paths_and_labels(root)
    train_idx, val_idx, test_idx = create_split_masks(root, train_ratio=0.7, val_ratio=0.2,
                                                         test_ratio=0.1, seed=0)

    # Training dataset
    training_paths = [audio_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    audio_augmentation_params = {
            'pitch_shift': [-3,3],
            'add_noise': [0.001, 0.1],
            'shift_max': 0.2,  # seconds
            'volume_gain_dB': [-4,4],
            'reverb': True
        }
    mfcc_augmentation_params = {
            'add_noise': [0.001, 0.01],
            'dropout': 0.1

        }
    train_dataset = AugmentedMFCCDataset(training_paths, train_labels, label_map,mixup=mixup,audio_augment=augment,audio_params=audio_augmentation_params,mfcc_augment=augment,mfcc_params=mfcc_augmentation_params, training=True)

    # Evaluation dataset
    validating_paths = [audio_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    val_dataset = AugmentedMFCCDataset(validating_paths, val_labels, label_map, training=False,mixup=mixup)
    # =====grid search loop=====
    for trial in range(20):
        # random model params
        num_epochs = 200#random.randint(150, 150)
        lr = 10 ** random.uniform(-6, -4)
        batch_size = 2 ** random.randint(4, 7)
        dropout = random.uniform(0, 0.2)
        # print('lr:', lr)
        # print('batch_size:', batch_size)
        # print('dropout:', dropout)
        lr=5.6325411498664184e-05
        batch_size = 32
        dropout = 0.17643
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        main_train_loop(train_loader, val_loader,mixup, num_epochs, lr, batch_size, dropout, device,
                        best_model_dir)
        print("trial number", trial)

    # test model
    testing_paths = [audio_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_loader = DataLoader(AugmentedMFCCDataset(testing_paths, test_labels, label_map, training=False), batch_size=1, shuffle=False)
    evaluate_model_on_test(test_loader, '../' + best_model_dir, device)


if __name__ == '__main__':
    grid_search()
