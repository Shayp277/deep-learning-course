import hashlib
import os
import random
from collections import defaultdict

import torch
import torch.nn.functional as torch_func
import torchaudio
from torch.utils.data import Dataset
from torch_audiomentations import Compose, Gain, Shift, PitchShift, ApplyImpulseResponse


class AugmentedMFCCDataset(Dataset):
    def __init__(self,
                 audio_paths,
                 labels=None,
                 label_map=None,
                 rir_dir=None,
                 sample_rate=16000,
                 audio_augment=False,
                 audio_params=None,
                 mfcc_augment=False,
                 mfcc_params=None,
                 training=False,
                 mixup=False,
                 classes_num=None,
                 is_multilabel=False,
                 drone_fine_tune=False,
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
        self.drone_fine_tune = drone_fine_tune
        self.mixup = mixup
        self.mixup_labels_soft = []
        self.mixup_labels_hard = []
        self.clean_mfcc = []
        self.augwav_mfcc = []
        self.mixup_mfcc = []
        self.aug_waveoforms = []
        self.final_input = []
        self.final_labels = []
        self.transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=80,
                                                    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128,
                                                               'center': True, 'power': 2.0})  # MFCC extractor

        self.classes_num = classes_num
        self.is_multilabel = is_multilabel
        self.labels_hot_one = (
        [torch_func.one_hot(torch.tensor(k).long(), num_classes=self.classes_num).float() for k in
         self.labels])

    def preprocess(self, seed):
        random.seed(seed)

        cache_dir = "../audio_cache"  # Efficient audio data loading
        os.makedirs(cache_dir, exist_ok=True)

        # =====Create clean (unaugmented) data=====
        # ===Efficient audio loading===
        waveforms = []
        random.seed(0)
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

            self.clean_mfcc.append(torch.mean(self.transform(waveform.squeeze()), axis=1).unsqueeze(
                0))  # ===Create clean (unaugmented) MFCC data===

        # =====Create augmented data=====
        if self.audio_augment:
            self.create_augmented_data(waveforms)

        if self.drone_fine_tune:
            self.final_input, self.final_labels = self.mixup_for_finetuning(seed)
        else:
            # =====Create Mixup data (multi-labeled)=====
            mix_waveforms = waveforms
            if self.mixup and self.training:
                p = 5 if self.training else 1
                while p > 0:
                    self.mixup_mfcc, self.mixup_labels_soft, self.mixup_labels_hard = self.create_mix(mix_waveforms)
                    p = p - 1
            if self.is_multilabel and not self.training:
                self.mixup_mfcc, self.mixup_labels_soft, self.mixup_labels_hard = self.create_mix(mix_waveforms)

            # Assign final data to dataset
            if self.training:
                self.final_input = self.clean_mfcc.copy()
                self.final_labels = self.labels_hot_one.copy()
                if self.is_multilabel and self.mixup:
                    self.final_input.extend(self.mixup_mfcc.copy())
                    self.final_labels.extend(self.mixup_labels_soft.copy())
                else:
                    if self.augwav_mfcc:
                        self.final_input.extend(self.augwav_mfcc.copy())
                        self.final_labels.extend(self.labels_hot_one.copy())
                    if self.mixup:
                        self.final_input.extend(self.mixup_mfcc.copy())
                        self.final_labels.extend(self.mixup_labels_soft.copy())
            else:
                if self.is_multilabel:
                    self.final_input.extend(self.mixup_mfcc.copy())
                    self.final_labels.extend(self.mixup_labels_hard.copy())
                else:
                    if not self.audio_augment:
                        self.final_input = self.clean_mfcc.copy()
                        self.final_labels = self.labels_hot_one.copy()
                    else:
                        self.final_input.extend(self.augwav_mfcc.copy())
                        self.final_labels.extend(self.labels_hot_one.copy())

    def create_augmented_data(self, waveforms):
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
            self.augwav_mfcc.append(torch.mean(self.transform(aug_waveform.squeeze()), axis=1).unsqueeze(0))

    def create_mix(self, mix_waveforms):
        mfcc_mix = []
        softlabel = []
        hardlabel = []
        for j in range(len(mix_waveforms)):
            i = j

            while self.labels[j] == self.labels[i]:
                i = random.randint(0, len(mix_waveforms) - 1)
            m = random.uniform(0.3, 0.7)
            min_audio_size = min(mix_waveforms[j].shape[1],
                                 mix_waveforms[i].shape[1])  # crop audio so it would be the same length
            mixup_waveforms = (m * mix_waveforms[j][:, :min_audio_size] + (1 - m) * mix_waveforms[i][:,
                                                                                    :min_audio_size])  # mix two audio files at random magnitude (sum of 1)
            # Keep mixup data (could be clean or augmented)
            mfcc_mix.append((torch.mean(self.transform(mixup_waveforms.squeeze()), axis=1).unsqueeze(0)))
            softlabel.append((m * self.labels_hot_one[j] + (1 - m) * self.labels_hot_one[i]))
            hardlabel.append((self.labels_hot_one[j] + self.labels_hot_one[i]))
        return mfcc_mix, softlabel, hardlabel

    @staticmethod
    def rms_normalize(waveform, target_db=-20):
        # convert target db to rms amplitude
        target_rms = 10 ** (target_db / 20)
        rms = waveform.pow(2).mean().sqrt()
        if rms > 0:
            waveform = waveform * (target_rms / rms)
        return waveform

    def mixup_for_finetuning(self, seed):

        # extract features from dataset with mixup

        random.seed(seed)
        mfcc_with_drone_mixup = []
        labels = []
        drone_file = []
        non_drone_file = []
        for path in self.audio_paths:
            if 'Drone' in path:
                drone_file.append(path)
            else:
                non_drone_file.append(path)

        for path in self.audio_paths:
            if 'Drone' in path:
                label = 1
            else:
                label = 0
            audio_origin, sample_rate = torchaudio.load(path)
            idx_non_drone = random.randint(0, len(non_drone_file) - 1)
            idx_drone = random.randint(0, len(drone_file) - 1)  # randomly select another audio file
            audio_mix, _ = torchaudio.load(non_drone_file[idx_non_drone])
            mix_up_norm = 0.5
            # mix_up_norm1 = random.uniform(0, 1)
            # mix_up_norm2 = random.uniform(0, 1-mix_up_norm1)
            # mix_up_norm3 = 1 - mix_up_norm1 - mix_up_norm2

            if random.random() > 0.5:
                label = 1
                audio_mix_drone, _ = torchaudio.load(drone_file[idx_drone])
                min_audio_size = min(audio_mix.shape[1], audio_origin.shape[1], audio_mix_drone.shape[1])  # crop audio
                audio = (self.rms_normalize(audio_origin[:, :min_audio_size])
                         + mix_up_norm * self.rms_normalize(audio_mix_drone[:, :min_audio_size]))
            else:
                min_audio_size = min(audio_mix.shape[1], audio_origin.shape[1], audio_mix.shape[1])
                audio = (self.rms_normalize(audio_origin[:, :min_audio_size])
                         + mix_up_norm * self.rms_normalize(audio_mix[:, :min_audio_size]))

            resample = torchaudio.transforms.Resample(sample_rate, sample_rate / 2)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdims=True)

            # extract features
            mfcc_with_drone_mixup.append(torch.mean(self.transform(resample(audio.squeeze())), axis=1).unsqueeze(dim=0))
            labels.append(label)
        labels_tensor = torch.tensor(labels)
        labels_hot_one = torch_func.one_hot(labels_tensor, num_classes=2).float()
        return mfcc_with_drone_mixup, labels_hot_one

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
        return len(self.final_labels)  # * factor

    def __getitem__(self, idx):
        """
        Part of DataLoader main function, design to query a sample.
        :param idx:
        :return: mfcc (input to nn) and label
        """
        if len(self.final_labels[idx]) == 0:
            raise ValueError(f"Empty label for index {idx}")
        mfcc = self.final_input[idx]
        label = self.final_labels[idx]

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
                mfcc = torch_func.dropout(mfcc, p=self.mfcc_params['dropout'], training=True)

        label_tensor = torch.tensor(label)
        return mfcc, label_tensor

    @staticmethod
    def get_cache_path(path, cache_dir):
        """Generate a unique file name using hash (to handle files with same name)."""
        name_hash = hashlib.md5(path.encode()).hexdigest()
        return os.path.join(cache_dir, f"{name_hash}.pt")


def create_split_masks(root_dir, train_ratio=None, val_ratio=None, test_ratio=None, seed=None):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    # Get number of audio samples in dataset
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
    class_names = (os.listdir(root_dir))
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


def create_split_masks_stratified(labels, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1"

    random.seed(seed)

    # Group indices by class
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        n = len(indices)

        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train_idx.extend(indices[:train_end])
        val_idx.extend(indices[train_end:val_end])
        test_idx.extend(indices[val_end:])

    # Shuffle the combined splits (optional, but keeps randomness)
    random.shuffle(train_idx)
    random.shuffle(val_idx)
    random.shuffle(test_idx)

    return train_idx, val_idx, test_idx
