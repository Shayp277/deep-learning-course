import os
import random

import torch
import torchaudio
from torch_audiomentations import (
    Gain, Shift, PitchShift, ApplyImpulseResponse, Compose
)


class PreMFCCAugment:
    def __init__(self, rir_dir=None, sample_rate=16000, params=None):
        self.sample_rate = sample_rate
        self.rir_dir = rir_dir
        self.params = params or {}

        self.augmentations = []

        # Time shift
        if 'shift_max' in self.params:
            self.augmentations.append(
                Shift(
                    min_shift=-self.params['shift_max'],
                    max_shift=self.params['shift_max'],
                    p=1.0,
                    output_type="tensor"
                )
            )

        # Gain
        if 'volume_gain_dB' in self.params:
            min_gain_db, max_gain_db = self.params['volume_gain_dB']
            self.augmentations.append(
                Gain(
                    min_gain_in_db=min_gain_db,
                    max_gain_in_db=max_gain_db,
                    p=1.0,
                    output_type="tensor"
                )
            )

        # Pitch shift (requires torch >= 1.9)
        if 'pitch_shift' in self.params:
            min_pitch_shift_db, max_pitch_shift_db = self.params['pitch_shift']
            self.augmentations.append(
                PitchShift(
                    sample_rate=self.sample_rate,
                    min_transpose_semitones=min_pitch_shift_db,
                    max_transpose_semitones=max_pitch_shift_db,
                    p=1.0,
                    output_type="tensor"
                )
            )

        # Reverb / impulse response
        if self.params.get('reverb') and self.rir_dir:
            self.augmentations.append(
                ApplyImpulseResponse(ir_paths=self._collect_rir_paths(self.rir_dir), p=1.0, output_type="tensor")
            )

        self.pipeline = Compose(self.augmentations, output_type="tensor")

    def _collect_rir_paths(self, rir_dir):
        rir_files = []
        for root, _, files in os.walk(rir_dir):
            for f in files:
                if f.lower().endswith('.wav'):
                    rir_files.append(os.path.join(root, f))
        if not rir_files:
            raise ValueError(f"No .wav RIR files found in {rir_dir}")
        return rir_files

    def __call__(self, audio_path, output_path):
        # Load and prepare waveform
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.unsqueeze(0)  # if waveform.dim() == 1 else waveform  # shape: [1, T] or [C, T]

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply augmentations
        waveform_aug = self.pipeline(waveform, sample_rate=self.sample_rate)  # shape: [1, 1, T]
        waveform_aug = waveform_aug.squeeze(0)  # shape: [1, T]

        # Add noise (manual, Gaussian)
        if 'add_noise' in self.params:
            min_noise_std, max_noise_std = self.params['add_noise']
            noise_std = random.uniform(min_noise_std, max_noise_std)
            noise = torch.randn_like(waveform_aug) * noise_std
            waveform_aug += noise

        # Ensure output directory exists
        os.makedirs(os.path.join(output_path, "tryfolder"), exist_ok=True)

        # Save augmented audio
        save_path = os.path.join(output_path, "tryfolder", "tryfile_augmented.wav")
        torchaudio.save(save_path, waveform_aug, self.sample_rate)
        print(f"Saved augmented audio to {save_path}")

        return waveform_aug
