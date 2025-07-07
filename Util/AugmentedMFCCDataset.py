import os
import random

import torch
import torchaudio
import hashlib

import torch.nn.functional as F
from scipy.signal import waveforms
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
                 training=True,
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

        # Define waveform-level augmentations
        # Time shift
        if 'shift_max' in self.audio_params:
            self.audio_augmentations.append(
                Shift(
                    min_shift=-self.audio_params['shift_max'],
                    max_shift=self.audio_params['shift_max'],
                    p=1.0,
                    output_type="tensor"
                )
            )

        # Gain
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

        # Pitch shift (requires torch >= 1.9)
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

        # Reverb / impulse response
        if self.audio_params.get('reverb') and self.rir_dir:
            self.audio_augmentations.append(
                ApplyImpulseResponse(ir_paths=self._collect_rir_paths(self.rir_dir), p=1.0, output_type="tensor")
            )

        self.augmenter = Compose(self.audio_augmentations, output_type="tensor")

        # Define MFCC extractor
        self.mfcc_transform = torchaudio.transforms.MFCC(n_mfcc=80,
                                                         melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128,
                                                                    'center': True, 'power': 2.0})

        self.mfcc = []
        self.clean_mfcc = []
        cache_dir = "../audio_cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Extract MFCC
        transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=80,
                                               melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128,
                                                          'center': True, 'power': 2.0})
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
        mixup_waveforms = []
        mixup_labels = []
        if mixup:
            for j in range(len(waveforms)):
                i = random.randint(0, len(waveforms) - 1)
                m = random.uniform(0, 1)
                min_audio_size = min(waveforms[j].shape[1], waveforms[i].shape[1])  # crop audio
                mixup_waveforms.append(m * waveforms[j][:, :min_audio_size] + (1 - m) * waveforms[i][:, :min_audio_size])
                mixup_labels.append(m * F.one_hot(torch.tensor(labels[j] ).long(), num_classes=8).float() + (1 - m) * F.one_hot(torch.tensor(labels[i]).long(), num_classes=8).float())


            waveforms = mixup_waveforms
            self.labels = mixup_labels

        for j in range(len(waveforms)):
            # Batchify for augmenter: [B, C, T]
            waveform = waveforms[j].unsqueeze(0)  # [1, 1, T]

            # Apply audio augmentations
            if self.audio_augment and self.training:
                self.clean_mfcc.append(torch.mean(transform(waveform.squeeze()), axis=1).unsqueeze(0))
                waveform = self.augmenter(waveform, sample_rate=self.sample_rate)
                if 'add_noise' in self.audio_params:
                    min_noise_std, max_noise_std = self.audio_params['add_noise']
                    noise_std = random.uniform(min_noise_std, max_noise_std)
                    waveform += torch.randn_like(waveform) * noise_std

            waveform = waveform.squeeze(0)  # [1, T]

            mfcc = transform(waveform.squeeze())  # [n_mfcc, time]
            mfcc = torch.mean(mfcc, axis=1).unsqueeze(0)  # [1, n_mfcc]
            self.mfcc.append(mfcc)

            if (j + 1) % 100 == 0:
                print(f"{j + 1} audio files processed.")


    def _collect_rir_paths(self, rir_dir):
        rir_files = []
        for root, _, files in os.walk(rir_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    rir_files.append(os.path.join(root, f))
        if not rir_files:
            raise ValueError(f"No .wav RIR files found in {rir_dir}")
        return rir_files


    def __len__(self):
        factor = 1
        if self.mfcc_augment or self.audio_augment:
            factor +=1
        if self.mixup:
            factor +=1
        return len(self.audio_paths) * factor



    def __getitem__(self, idx):
        # Load audio
        # if self.mfcc_augment or self.audio_augment:
        real_idx = idx % len(self.labels)
        if idx < len(self.labels):
            mfcc = self.mfcc[real_idx]
        else:
            mfcc = self.clean_mfcc[real_idx]
        label = self.labels[real_idx]

        # mfcc = self.mfcc[idx]

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

        if self.mixup:
            label_tensor = torch.tensor(label)
        else:
            # label_tensor = F.one_hot(torch.tensor(label).long(), num_classes=8).float()  # [num_classes]
            label_tensor = torch.tensor(label)
        return mfcc, label_tensor

        # mfcc = torch.mean(mfcc, axis=2)

        # label = self.labels[idx]


    @staticmethod
    def get_cache_path(path, cache_dir):
        """Generate a unique file name using hash (to handle files with same name)."""
        name_hash = hashlib.md5(path.encode()).hexdigest()
        return os.path.join(cache_dir, f"{name_hash}.pt")
