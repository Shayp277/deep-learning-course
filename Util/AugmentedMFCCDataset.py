import os
import random

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_audiomentations import Compose, Gain, Shift, PitchShift, ApplyImpulseResponse

class AugmentedMFCCDataset(Dataset):
    def __init__(self,
                 audio_paths,
                 labels=None,
                 label_map = None,
                 rir_dir=None,
                 sample_rate=16000,
                 n_mfcc=80,
                 audio_augment=False,
                 audio_params=None,
                 mfcc_augment=False,
                 mfcc_params=None,
                 training=True
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
        self.mfcc_transform = torchaudio.transforms.MFCC(sample_rate=int(sample_rate/2), n_mfcc=80, melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 128, 'center': True, 'power': 2.0})

        # Define MFCC Augmentation

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
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # Load audio
        audio_path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(audio_path)
        # print("Waveform shape when loaded:", waveform.shape)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        # print("Waveform shape after sr correction:", waveform.shape)
        # waveform = waveform.unsqueeze(0)  # if waveform.dim() == 1 else waveform  # shape: [1, T] or [C, T]
        # print("Waveform shape after unsqueeze (1,T):", waveform.shape)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # print("Waveform shape after mono:", waveform.shape)
        # Batchify for augmenter: [B, C, T]
        waveform = waveform.unsqueeze(0)  # shape [1, 1, T]
        # print("Waveform shape after unsqueeze:", waveform.shape)
        # Apply audio augmentations
        if self.audio_augment and self.training:
            waveform_aug = self.augmenter(waveform, sample_rate=self.sample_rate) # shape: [1, 1, T]
            if 'add_noise' in self.audio_params:
                min_noise_std, max_noise_std = self.audio_params['add_noise']
                noise_std = random.uniform(min_noise_std, max_noise_std)
                noise = torch.randn_like(waveform_aug) * noise_std
                waveform_aug += noise
        waveform = waveform.squeeze(0)  # back to [1, T]
        # print("Waveform shape after squeeze:", waveform.shape)
        # Extract MFCC
        mfcc = self.mfcc_transform(waveform)  # shape: [n_mfcc, time]
        # print("mfcc shape:", mfcc.shape)
        # Post-MFCC augmentations (only in training)
        if self.mfcc_augment and self.training:
            # Add Gaussian noise
            if 'add_noise' in self.mfcc_params:
                noise = torch.randn_like(mfcc) * self.mfcc_params['noise_std']
                mfcc = mfcc + noise
            # Apply dropout
            if 'dropout' in self.mfcc_params:
                mfcc = F.dropout(mfcc, p=self.mfcc_params['dropout_p'], training=True)

        mfcc = torch.mean(mfcc, axis=1)
        T = mfcc.shape[1]
        max_frames = 5*self.sample_rate
        if T > max_frames:
            return mfcc[:, :max_frames]
        else:
            mfcc =  torch.nn.functional.pad(mfcc, (0, max_frames - T))
        # print("final mfcc shape:", mfcc.shape)
        if self.labels:
            label = self.labels[idx]  # e.g., "Cars"
            label_tensor = F.one_hot(torch.tensor(label).long(), num_classes=8).float()  # [num_classes]
            return mfcc, label_tensor
        else:
            return mfcc