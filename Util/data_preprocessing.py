import os
import pickle
import torch
import pandas as pd
import torchaudio
import random
import fnmatch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import IPython.display as idp


class DATA:
    def __init__(self, audio_dataset_path, wav_to_pkl=False, with_mixup=False):
        # variables
        self.data = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.X_Val = []
        self.y_Val = []
        self.X_test_with_mixup = []
        self.y_test_with_mixup = []

        # wav to pkl variables
        self.extracted_features_train = []
        self.extracted_features_val = []
        self.extracted_features_test = []

        # split file name to datasets
        self.train_file_name = []
        self.val_file_name = []
        self.test_file_name = []

        self.train_mask, self.val_mask, self.test_mask = self.create_split_masks(self.get_data_size(audio_dataset_path))
        # extract and save features from audio files
        if wav_to_pkl:
            idx = 0
            for path in os.listdir(audio_dataset_path):
                for file in os.listdir(audio_dataset_path + "/" + path + "/"):
                    file_name = audio_dataset_path + "/" + path + "/" + file
                    # data = self.features_extractor(file_name)
                    # self.extracted_features.append([data, path])
                    if self.train_mask[idx]:
                        data = self.features_extractor(file_name)
                        self.extracted_features_train.append([data, path])
                        self.train_file_name.append(file_name)
                    elif self.val_mask[idx]:
                        data = self.features_extractor(file_name)
                        self.extracted_features_val.append([data, path])
                        self.val_file_name.append(file_name)
                    elif self.test_mask[idx]:
                        data = self.features_extractor(file_name)
                        self.extracted_features_test.append([data, path])
                        self.test_file_name.append(file_name)
                    idx += 1

            self.save_to_pkl(self.extracted_features_train, 'train')
            self.save_to_pkl(self.extracted_features_val, 'validation')
            self.save_to_pkl(self.extracted_features_test, 'test')

            self.extracted_features_train = self.features_extractor_with_mixup(self.train_file_name)
            self.extracted_features_val = self.features_extractor_with_mixup(self.val_file_name)
            self.extracted_features_test = self.features_extractor_with_mixup(self.test_file_name)

            self.save_to_pkl(self.extracted_features_train, 'train_with_mixup')
            self.save_to_pkl(self.extracted_features_val, 'val_with_mixup')
            self.save_to_pkl(self.extracted_features_test, 'test_with_mixup')

        # choose which dataset to use, in both cases evaluation will be performed on the test set, with and without MixUp applied.
        if with_mixup:
            self.X_train, self.y_train = self.load_from_pkl('train_with_mixup')
            self.X_val, self.y_val = self.load_from_pkl('val_with_mixup')
        else:
            self.X_train, self.y_train = self.load_from_pkl('train')
            self.X_val, self.y_val = self.load_from_pkl('validation')

        self.X_test, self.y_test = self.load_from_pkl('test')
        self.X_test_with_mixup, self.y_test_with_mixup = self.load_from_pkl('test_with_mixup')

    def get_data_size(self, audio_dataset_path):
        """
        return number of audio samples in dataset
        :param audio_dataset_path:
        :return: dataset size
        """
        data_size = 0
        for path in os.listdir(audio_dataset_path):
            data_size += len(os.listdir(audio_dataset_path + "/" + path + "/"))
        return data_size

    def save_to_pkl(self, data, file_name):
        """
        save data to pkl
        :param data:
        :param file_name:
        """
        f = open(os.path.join('../PKLS', file_name + '.pkl'), 'wb')
        pickle.dump(data, f)
        f.close()

    def load_from_pkl(self, file_name):
        """
        load data from pkl, convert data to torch tensor
        :param file_name:
        :return: save data to variables
        """
        f = open(os.path.join('../PKLS', file_name + '.pkl'), 'rb')
        self.data = pickle.load(f)
        f.close()
        # convert data to torch tensor
        self.df = self.to_dataframe()
        return self.dataframe_to_tensor()

    def to_dataframe(self):
        """
        convert data to dataframe
        :return: dataframe
        """
        return pd.DataFrame(self.data, columns=['feature', 'class'])

    def features_extractor(self, dataset_path):
        """
        extract features from dataset
        :param dataset_path:
        :return: features size([1 1 80])
        """
        audio, sample_rate = torchaudio.load(dataset_path)
        resample = torchaudio.transforms.Resample(sample_rate, sample_rate/2)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdims=True)

        # extract features
        transform = torchaudio.transforms.MFCC(sample_rate=sample_rate/2, n_mfcc=80, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128, 'center': True, 'power': 2.0})
        mfcc = transform(resample(audio.squeeze()))
        return torch.mean(mfcc, axis=1).unsqueeze(dim=0)

    def features_extractor_with_mixup(self, file_names):
        """
        extract features from dataset with mixup
        :param file_names:
        :return: features size([1 1 80])
        """
        random.seed(0)
        extracted_features_with_mixup = []
        for file in file_names:
            audio_origin, sample_rate = torchaudio.load(file)
            idx = random.randint(0, len(file_names) - 1)                         # randomly select another audio file
            audio_mix, _ = torchaudio.load(file_names[idx])
            min_audio_size = min(audio_mix.shape[1], audio_origin.shape[1])                   # crop audio
            audio = audio_origin[:,:min_audio_size] + 0.5 * audio_mix[:,:min_audio_size]      # mixup
            resample = torchaudio.transforms.Resample(sample_rate, sample_rate/2)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdims=True)

            # extract features
            transform = torchaudio.transforms.MFCC(sample_rate=sample_rate/2, n_mfcc=80, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128, 'center': True, 'power': 2.0})
            mfcc = transform(resample(audio.squeeze()))
            extracted_features_with_mixup.append([torch.mean(mfcc, axis=1).unsqueeze(dim=0), file.split('/')[2]])
        return extracted_features_with_mixup

    def dataframe_to_tensor(self):
        X = torch.stack(self.df['feature'].tolist())
        y = self.df['class'].tolist()

        # Represent labels as one-hot vector
        labelencoder = LabelEncoder()
        y = torch.nn.functional.one_hot(torch.from_numpy(labelencoder.fit_transform(y)).long(),
                                        num_classes=self.df['class'].unique().size).float()

        return X, y

    def create_split_masks(self, dataset_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=0):
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

        # Create boolean masks
        train_mask = torch.zeros(dataset_size, dtype=torch.bool)
        val_mask = torch.zeros(dataset_size, dtype=torch.bool)
        test_mask = torch.zeros(dataset_size, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask