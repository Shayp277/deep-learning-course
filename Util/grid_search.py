from Util.test import *
from Util.train import *
import numpy as np

random.seed(0)


def grid_search():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_dir = 'shay_best_model'  # choose dir to save current best model
    # download_data()                                                                 # if you need to create new dataset you need to download wav file
    # data = DATA('../data', wav_to_pkl=False, with_mixup=True)        # to create new pkl files choose wav_to_pkl=True, for loading data with mixup choose with_mixup=True
    root = "../data"
    audio_paths, labels, label_map = load_audio_paths_and_labels(root)
    train_idx, val_idx, test_idx = create_split_masks(get_data_size(root), train_ratio=0.7, val_ratio=0.2,
                                                         test_ratio=0.1, seed=0)

    # Training dataset
    training_paths = [audio_paths[i] for i in train_idx]
    train_dataset = AugmentedMFCCDataset(training_paths, labels, label_map, training=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Evaluation dataset
    validating_paths = [audio_paths[i] for i in val_idx]
    val_dataset = AugmentedMFCCDataset(validating_paths, labels, label_map, training=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)

    for trail in range(10):
        # random model params
        num_epochs = random.randint(150, 150)
        lr = 10 ** random.uniform(-5, -3)
        batch_size = 2 ** random.randint(5, 8)
        dropout = random.uniform(0, 0.2)

        # train
        main_train_loop(train_loader, val_loader, num_epochs, lr, batch_size, dropout, device,
                        best_model_dir)

    # test model
    testing_paths = [audio_paths[i] for i in test_idx]
    test_loader = DataLoader(AugmentedMFCCDataset(testing_paths, labels, label_map, training=False), batch_size=1, shuffle=False)
    evaluate_model_on_test(test_loader, '../' + best_model_dir, device)
    evaluate_model_on_test(test_loader, '../' + best_model_dir, device)
    # To view the training progress for all models in a given directory, enter the following command in the PyCharm terminal:
    #       tensorboard --logdir=<Enter your result dir>
    # example:
    #       tensorboard --logdir=runs


def create_split_masks(dataset_size, train_ratio=None, val_ratio=None, test_ratio=None, seed=None):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random permutation of indices
    indices = np.random.permutation(dataset_size)

    # Compute split sizes
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    # Create index splits
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # # Create boolean masks
    # train_mask = torch.zeros(dataset_size, dtype=torch.bool)
    # val_mask = torch.zeros(dataset_size, dtype=torch.bool)
    # test_mask = torch.zeros(dataset_size, dtype=torch.bool)
    #
    # train_mask[train_idx] = True
    # val_mask[val_idx] = True
    # test_mask[test_idx] = True

    return train_idx, val_idx, test_idx


def get_data_size(audio_dataset_path):
    """
    return number of audio samples in dataset
    :param audio_dataset_path:
    :return: dataset size
    """
    data_size = 0
    for path in os.listdir(audio_dataset_path):
        data_size += len(os.listdir(audio_dataset_path + "/" + path + "/"))
    return data_size


if __name__ == '__main__':
    grid_search()
