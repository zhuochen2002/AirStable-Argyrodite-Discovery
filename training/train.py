"""Main training entry: dataloaders, model, Trainer."""
import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import MODEL_CONFIG, TRAIN_CONFIG, LOSS_TYPE, DATA_PATH
from data.dataset import FormationEnergyDataset, get_element_vocabulary_from_mat2vec
from models.model import FormationEnergyModel
from utils.mat2vec import load_embeddings_matrix
from training.trainer import Trainer


def collate_fn(batch):
    """Stack batch dicts from FormationEnergyDataset."""
    element_indices = torch.stack([item['element_indices'] for item in batch])
    fractions = torch.stack([item['fractions'] for item in batch])
    mask = torch.stack([item['mask'] for item in batch])
    formation_energy = torch.stack([item['formation_energy'] for item in batch])
    return {
        'element_indices': element_indices,
        'fractions': fractions,
        'mask': mask,
        'formation_energy': formation_energy,
    }


def main():
    print("=" * 60)
    print("Formation Energy Prediction - Training")
    print("=" * 60)
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()

    print("Loading datasets...")
    train_data = pd.read_csv("data/train.csv")
    val_data = pd.read_csv("data/val.csv")
    print(f"  Train samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print()

    print("Loading element vocabulary from mat2vec...")
    element_to_idx = get_element_vocabulary_from_mat2vec()
    print()

    print("Creating datasets...")
    train_dataset = FormationEnergyDataset(
        data=train_data,
        element_to_idx=element_to_idx,
        n_max=MODEL_CONFIG['n_max']
    )
    val_dataset = FormationEnergyDataset(
        data=val_data,
        element_to_idx=element_to_idx,
        n_max=MODEL_CONFIG['n_max']
    )
    print()

    print("Creating data loaders...")
    num_workers = TRAIN_CONFIG.get('num_workers', 0)
    pin_memory = TRAIN_CONFIG.get('pin_memory', True) if torch.cuda.is_available() else False
    prefetch_factor = TRAIN_CONFIG.get('prefetch_factor', 2)
    persistent_workers = TRAIN_CONFIG.get('persistent_workers', True) if num_workers > 0 else False

    print(f"  DataLoader config:")
    print(f"    num_workers: {num_workers}")
    print(f"    pin_memory: {pin_memory}")
    print(f"    prefetch_factor: {prefetch_factor}")
    print(f"    persistent_workers: {persistent_workers}")
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print()

    print("Loading mat2vec embeddings...")
    embeddings_matrix, _, _ = load_embeddings_matrix()
    mat2vec_tensor = torch.from_numpy(embeddings_matrix).float()
    print(f"  Embeddings shape: {mat2vec_tensor.shape}")
    print()

    print("Building model...")
    model = FormationEnergyModel(MODEL_CONFIG, mat2vec_tensor)
    print(f"  Model parameters: {model.count_parameters():,}")
    print()

    print("Creating trainer...")
    trainer_config = {
        **TRAIN_CONFIG,
        'checkpoint_dir': 'checkpoints'
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device,
        loss_type=LOSS_TYPE
    )
    print()

    trainer.train()

    print("\nTraining completed successfully!")
    print(f"Best model saved to: checkpoints/best_model.pt")


if __name__ == "__main__":
    main()
