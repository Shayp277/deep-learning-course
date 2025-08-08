import os
import random

import torch
import torchaudio
import hashlib

import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_audiomentations import Compose, Gain, Shift, PitchShift, ApplyImpulseResponse


class AugmentedMFCCDataset(Dataset):
    def __init__(self,
                 audio_paths,
                 labels=None,
                 label_map=None,
                 rir_dir=None,
                 sample_rate=16000,
                 n_mfcc=80,
                 audio_augment=False,
                 audio_params=None,
                 mfcc_augment=False,
                 mfcc_params=None,
                 training=False,
                 mixup=False
                 ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.label_map = label_map
        self.rir_dir = rir_dir
        self.sample_rate = sample_rate
        self.audio_augment = audio_augment
        self.audio_params = audio_params or {}
        self.mfcc_augment = mfcc_augment
        self.mfcc_params = mfcc_params or {}
        self.audio_augmentations = []
        self.augmenter = []
        self.training = training
        self.mixup = mixup

        # =====Define waveform-level augmentations=====
        # ===Time shift===
        if 'shift_max' in self.audio_params:
            self.audio_augmentations.append(
                Shift(
                    min_shift=-self.audio_params['shift_max'],
                    max_shift=self.audio_params['shift_max'],
                    p=1.0,
                    output_type="tensor"
                )
            )

        # ===Gain===
        if 'volume_gain_dB' in self.audio_params:
            min_gain_db, max_gain_db = self.audio_params['volume_gain_dB']
            self.audio_augmentations.append(
                Gain(
                    min_gain_in_db=min_gain_db,
                    max_gain_in_db=max_gain_db,
                    p=1.0,
                    output_type="tensor"
                )
            )

        # ===Pitch shift===
        if 'pitch_shift' in self.audio_params:
            min_pitch_shift_db, max_pitch_shift_db = self.audio_params['pitch_shift']
            self.audio_augmentations.append(
                PitchShift(
                    sample_rate=self.sample_rate,
                    min_transpose_semitones=min_pitch_shift_db,
                    max_transpose_semitones=max_pitch_shift_db,
                    p=1.0,
                    output_type="tensor"
                )
            )

        # ===Reverb / impulse response===
        if 'reverb' in self.audio_params and self.rir_dir:
            self.audio_augmentations.append(
                ApplyImpulseResponse(ir_paths=self._collect_rir_paths(self.rir_dir), p=1.0, output_type="tensor")
            )

        self.augmenter = Compose(self.audio_augmentations, output_type="tensor")


        # self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=80,
        #                                                  melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128,
        #                                                             'center': True, 'power': 2.0})


        self.clean_mfcc = []
        self.augwav_mfcc = []
        self.mixup_mfcc = []
        self.aug_waveoforms = []
        self.options = [] #used at __get_item__ for using multiple data types (clean, augmented, mixed)
        self.options.append('clean')
        cache_dir = "../audio_cache" #Efficient audio data loading
        os.makedirs(cache_dir, exist_ok=True)

        # =====MFCC extractor=====
        transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=80,
                                               melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128,
                                                          'center': True, 'power': 2.0})
        # =====Create clean (unaugmented) data=====
        # ===Efficient audio loading===
        waveforms = []
        for j, path in enumerate(self.audio_paths):
            cache_path = self.get_cache_path(path, cache_dir)

            # Load or preprocess and save
            if os.path.exists(cache_path):
                waveform = torch.load(cache_path)
            else:
                waveform, sr = torchaudio.load(path)
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                torch.save(waveform, cache_path)
            waveforms.append(waveform)

            self.clean_mfcc.append(torch.mean(transform(waveform.squeeze()), axis=1).unsqueeze(0)) # ===Create clean (unaugmented) MFCC data===

        # =====Create augmented data=====
        if self.audio_augment and self.training:
            self.options.append('augment')
            for j in range(len(waveforms)):

                aug_waveform = waveforms[j].unsqueeze(0)  # Batchify for augmenter: [B, C, T]

                # Apply audio augmentations
                aug_waveform = self.augmenter(aug_waveform, sample_rate=self.sample_rate)
                if 'add_noise' in self.audio_params:
                    min_noise_std, max_noise_std = self.audio_params['add_noise']
                    noise_std = random.uniform(min_noise_std, max_noise_std)
                    aug_waveform += torch.randn_like(aug_waveform) * noise_std
                aug_waveform = aug_waveform.squeeze(0)  # [1, T]

                # Keep augmented data
                self.aug_waveoforms.append(aug_waveform)
                self.augwav_mfcc.append(torch.mean(transform(aug_waveform.squeeze()), axis=1).unsqueeze(0))

                if (j + 1) % 100 == 0:
                    print(f"{j + 1} audio files processed.")

        # =====Create Mixup data (multi-labeled)=====
        self.mixup_labels = []
        if mixup:
            self.options.append('mixup')
            if self.audio_augment:
                mix_waveforms = self.aug_waveoforms
            else:
                mix_waveforms = waveforms
            self.labels = ([F.one_hot(torch.tensor(k).long(), num_classes=8).float() for k in self.labels]) #while doing multi- labeling, labels should be in a vector form and not an index, e.g [0,0.2,0,0.8.0]
            for j in range(len(mix_waveforms)):
                i = random.randint(0, len(mix_waveforms) - 1)
                m = random.uniform(0, 1)
                min_audio_size = min(mix_waveforms[j].shape[1], mix_waveforms[i].shape[1])  # crop audio so it would be the same length
                mixup_waveforms = (m * mix_waveforms[j][:, :min_audio_size] + (1 - m) * mix_waveforms[i][:, :min_audio_size]) #mix two audio files at random magnitude (sum of 1)
                # Keep mixup data (could be clean or augmented)
                self.mixup_mfcc.append(torch.mean(transform(mixup_waveforms.squeeze()), axis=1).unsqueeze(0))
                self.mixup_labels.append(m * self.labels[j] + (1 - m) * self.labels[i])

                if (j + 1) % 100 == 0:
                    print(f"{j + 1} audio files processed.")

        self.Final_input = self.clean_mfcc.copy()
        self.Final_labels = self.labels.copy()
        if self.augwav_mfcc:
            self.Final_input.extend(self.augwav_mfcc)
            self.Final_labels.extend(self.labels)
        if self.mixup_mfcc:
            self.Final_input.extend(self.mixup_mfcc)
            self.Final_labels.extend(self.mixup_labels)



    @staticmethod
    def _collect_rir_paths(rir_dir):
        """
        Used to load rir files for the reverb (fir impulse response) augmentation.
        :param rir_dir:
        :return:
        """
        rir_files = []
        for root, _, files in os.walk(rir_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    rir_files.append(os.path.join(root, f))
        if not rir_files:
            raise ValueError(f"No .wav RIR files found in {rir_dir}")
        return rir_files


    def __len__(self):
        """
        Return the number of audio files in the dataset. Depends on whether using mixup or augmentation.
        """
        # factor = 1
        # if self.mfcc_augment or self.audio_augment:
        #     factor +=1
        # if self.mixup:
        #     factor +=1
        return len(self.Final_labels) #* factor



    def __getitem__(self, idx):
        """
        Part of DataLoader main function, design to query a sample.
        :param idx:
        :return: mfcc (input to nn) and label
        """
        # Load (input, target output) tuple randomly.
        # choice = random.choice(self.options)
        # real_idx = idx % len(self.audio_paths)
        # if choice == 'clean':
        #     mfcc = self.clean_mfcc[real_idx]
        #     label = self.labels[real_idx]
        # if choice == 'mixup':
        #     mfcc = self.mixup_mfcc[real_idx]
        #     label = self.mixup_labels[real_idx]
        # if choice == 'augment':
        #     mfcc = self.augwav_mfcc[real_idx]
        #     label = self.labels[real_idx]
        mfcc = self.Final_input[idx]
        label = self.Final_labels[idx]


        # Post-MFCC augmentations (only in training)
        if self.mfcc_augment and self.training:
            # Add Gaussian noise
            if 'add_noise' in self.mfcc_params:
                min_noise_std, max_noise_std = self.mfcc_params['add_noise']
                noise_std = random.uniform(min_noise_std, max_noise_std)
                noise = torch.randn_like(mfcc) * noise_std
                mfcc = mfcc + noise
            # Apply dropout
            if 'dropout' in self.mfcc_params:
                mfcc = F.dropout(mfcc, p=self.mfcc_params['dropout'], training=True)

        label_tensor = torch.tensor(label)
        return mfcc, label_tensor

    @staticmethod
    def get_cache_path(path, cache_dir):
        """Generate a unique file name using hash (to handle files with same name)."""
        name_hash = hashlib.md5(path.encode()).hexdigest()
        return os.path.join(cache_dir, f"{name_hash}.pt")

def create_split_masks(root_dir, train_ratio=None, val_ratio=None, test_ratio=None, seed=None):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    #Get number of audio samples in dataset
    dataset_size = 0
    for path in os.listdir(root_dir):
        dataset_size += len(os.listdir(root_dir + "/" + path + "/"))

    # Set random seed for reproducibility
    random.seed(seed)

    # Generate random permutation of indices
    indices = list(range(dataset_size))
    random.shuffle(indices)

    # Compute split sizes
    train_end = int(train_ratio * dataset_size)
    val_end = train_end + int(val_ratio * dataset_size)

    # Create index splits
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return train_idx, val_idx, test_idx

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