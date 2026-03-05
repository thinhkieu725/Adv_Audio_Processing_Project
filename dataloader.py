import os
import zipfile
import random
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BEATsDataset(Dataset):
    """
    Custom Dataset for BEATs hierarchical audio classification.
    """
    def __init__(self, file_list, child_labels, parent_labels, orig_sr=44100, target_sr=16000, duration_sec=10.0):
        self.file_list = file_list
        self.child_labels = child_labels
        self.parent_labels = parent_labels
        self.target_sr = target_sr
        self.resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        self.target_samples = int(target_sr * duration_sec)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        child_label = self.child_labels[idx]
        parent_label = self.parent_labels[idx]

        # Load and resample
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.target_sr:
            waveform = self.resampler(waveform)

        current_samples = waveform.shape[1]

        # Initialize the padding mask with all False (real audio)
        padding_mask = torch.zeros((1, self.target_samples), dtype=torch.bool)

        # 1. If larger: Randomly crop a 10-second subsequence
        if current_samples > self.target_samples:
            max_start = current_samples - self.target_samples
            start_idx = random.randint(0, max_start)
            waveform = waveform[:, start_idx : start_idx + self.target_samples]

        # 2. If smaller: Zero-pad at the end
        elif current_samples < self.target_samples:
            pad_amount = self.target_samples - current_samples
            waveform = F.pad(waveform, (0, pad_amount), mode='constant', value=0.0)
            padding_mask[:, current_samples:] = True

        # Explicitly enforce exact dimensions
        waveform = waveform[:, :self.target_samples]
        padding_mask = padding_mask[:, :self.target_samples]

        return waveform, padding_mask, parent_label, child_label


def prepare_dataframes(clean_csv_dir, noisy_csv_path, clean_audio_dir, noisy_audio_dir):
    """
    Loads CSVs, generates parent mapping, and merges Clean + Noisy for training.
    """
    clean_train_csv = os.path.join(clean_csv_dir, 'bsd10k-train.csv')
    clean_val_csv = os.path.join(clean_csv_dir, 'bsd10k-val.csv')
    clean_test_csv = os.path.join(clean_csv_dir, 'bsd10k-test.csv')

    df_train = pd.read_csv(clean_train_csv)
    df_val = pd.read_csv(clean_val_csv)
    df_test = pd.read_csv(clean_test_csv)
    df_noisy = pd.read_csv(noisy_csv_path)

    # Map labels to indices
    unique_parents = df_train['class_top'].unique()
    parent_name_to_id = {name: idx for idx, name in enumerate(unique_parents)}
    
    df_train['parent_idx'] = df_train['class_top'].map(parent_name_to_id)
    df_val['parent_idx'] = df_val['class_top'].map(parent_name_to_id)
    df_test['parent_idx'] = df_test['class_top'].map(parent_name_to_id)

    child_to_parent_dict = dict(zip(df_train['class_idx'], df_train['parent_idx']))
    df_noisy['parent_idx'] = df_noisy['class_idx'].map(child_to_parent_dict)

    if df_noisy['parent_idx'].isnull().any():
        df_noisy = df_noisy.dropna(subset=['parent_idx'])

    # Convert to strict integers
    df_train['parent_idx'] = df_train['parent_idx'].astype(int)
    df_val['parent_idx'] = df_val['parent_idx'].astype(int)
    df_test['parent_idx'] = df_test['parent_idx'].astype(int)
    df_noisy['parent_idx'] = df_noisy['parent_idx'].astype(int)

    # Generate file path columns
    df_train['file_path'] = os.path.join(clean_audio_dir, '') + df_train['sound_id'].astype(str) + '.wav'
    df_val['file_path'] = os.path.join(clean_audio_dir, '') + df_val['sound_id'].astype(str) + '.wav'
    df_test['file_path'] = os.path.join(clean_audio_dir, '') + df_test['sound_id'].astype(str) + '.wav'
    df_noisy['file_path'] = os.path.join(noisy_audio_dir, '') + df_noisy['sound_id'].astype(str) + '.wav'

    # Merge Clean + Noisy for training
    df_final_train = pd.concat([df_train, df_noisy], ignore_index=True)

    return df_final_train, df_val, df_test


def Dataloaders(clean_csv_dir, noisy_csv_path, clean_audio_dir, noisy_audio_dir, batch_size=16):
    """
    Constructs and returns PyTorch DataLoaders for Train, Val, and Test.
    """
    df_train, df_val, df_test = prepare_dataframes(
        clean_csv_dir, noisy_csv_path, clean_audio_dir, noisy_audio_dir
    )

    # Instantiate Datasets
    train_dataset = BEATsDataset(
        file_list=df_train['file_path'].tolist(),
        child_labels=df_train['class_idx'].tolist(),
        parent_labels=df_train['parent_idx'].tolist()
    )

    val_dataset = BEATsDataset(
        file_list=df_val['file_path'].tolist(),
        child_labels=df_val['class_idx'].tolist(),
        parent_labels=df_val['parent_idx'].tolist()
    )

    test_dataset = BEATsDataset(
        file_list=df_test['file_path'].tolist(),
        child_labels=df_test['class_idx'].tolist(),
        parent_labels=df_test['parent_idx'].tolist()
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def extract_dataset(zip_path, target_dir):
    """Utility to extract a zip file if needed before training."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"Extracting {os.path.basename(zip_path)} to {target_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    file_count = sum([len(files) for r, d, files in os.walk(target_dir)])
    print(f"Extracted {file_count} files.")
    return file_count
