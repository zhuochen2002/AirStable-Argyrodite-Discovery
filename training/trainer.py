"""Training loop, validation, early stopping, LR schedule, checkpoints, history JSON."""
import os
import sys
import json
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


class Trainer:
    """Fit FormationEnergyModel with AdamW, optional warmup+cosine LR, early stopping on val loss."""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device,
                 loss_type: str = "MAE"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.loss_type = loss_type

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.use_warmup = config.get('use_warmup', False)
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.lr_scheduler_type = config.get('lr_scheduler', 'cosine')
        self.lr_min = config.get('lr_min', 1e-6)
        self.base_lr = config['learning_rate']
        self.num_epochs = config['num_epochs']

        if self.use_warmup and self.warmup_epochs > 0:
            self.scheduler = self._create_warmup_cosine_scheduler()
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=self.lr_min
            )

        if loss_type == "MAE":
            self.criterion = nn.L1Loss()
        elif loss_type == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_type == "Huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'MAE', 'MSE', or 'Huber'")

        self.early_stopping_patience = config.get('early_stopping_patience', 20)
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.patience_counter = 0
        self.best_model_state = None

        self.train_losses = []
        self.train_maes = []
        self.val_losses = []
        self.val_maes = []

        self.checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_json_path = os.path.join(self.checkpoint_dir, 'training_history.json')

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0

        for batch in self.train_loader:
            element_indices = batch['element_indices'].to(self.device, non_blocking=True)
            fractions = batch['fractions'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            formation_energy = batch['formation_energy'].to(self.device, non_blocking=True).squeeze(-1)

            self.optimizer.zero_grad()
            predictions = self.model(element_indices, fractions, mask)
            loss = self.criterion(predictions, formation_energy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            with torch.no_grad():
                mae = torch.abs(predictions - formation_energy).mean().item()

            batch_size = element_indices.shape[0]
            total_loss += loss.item() * batch_size
            total_mae += mae * batch_size
            n_samples += batch_size

        return total_loss / n_samples, total_mae / n_samples

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                element_indices = batch['element_indices'].to(self.device, non_blocking=True)
                fractions = batch['fractions'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                formation_energy = batch['formation_energy'].to(self.device, non_blocking=True).squeeze(-1)

                predictions = self.model(element_indices, fractions, mask)
                loss = self.criterion(predictions, formation_energy)
                mae = torch.abs(predictions - formation_energy).mean().item()

                batch_size = element_indices.shape[0]
                total_loss += loss.item() * batch_size
                total_mae += mae * batch_size
                n_samples += batch_size

        return total_loss / n_samples, total_mae / n_samples

    def train(self):
        print("=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Loss type: {self.loss_type}")
        print("=" * 60)
        print()

        start_time = time.time()
        best_epoch = 0
        last_epoch = 0

        for epoch in range(1, self.config['num_epochs'] + 1):
            last_epoch = epoch
            epoch_start_time = time.time()

            train_loss, train_mae = self.train_epoch()
            val_loss, val_mae = self.validate()

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_losses.append(train_loss)
            self.train_maes.append(train_mae)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)

            self.save_training_history()

            epoch_time = time.time() - epoch_start_time
            warmup_indicator = " [WARMUP]" if self.use_warmup and epoch <= self.warmup_epochs else ""
            print(f"Epoch {epoch:3d}/{self.config['num_epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f} | "
                  f"LR: {current_lr:.2e}{warmup_indicator} | Time: {epoch_time:.2f}s")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_mae = val_mae
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
                best_epoch = epoch

                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                epoch_best_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
                self.save_checkpoint(epoch, best_path)
                self.save_checkpoint(epoch, epoch_best_path)

                print(f"  New best validation loss: {val_loss:.6f} (MAE: {val_mae:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    print(f"Best validation loss: {self.best_val_loss:.6f} (MAE: {self.best_val_mae:.6f})")
                    break

            print()

        total_time = time.time() - start_time
        print("=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Best validation MAE: {self.best_val_mae:.6f}")
        print(f"Best model saved to: {os.path.join(self.checkpoint_dir, 'best_model.pt')}")
        print(f"Training history saved to: {self.history_json_path}")
        print("=" * 60)

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Compatibility: some inference scripts default to `best_model_final.pt`.
        final_best_path = os.path.join(self.checkpoint_dir, 'best_model_final.pt')
        final_epoch = best_epoch if best_epoch > 0 else last_epoch
        if final_epoch > 0:
            self.save_checkpoint(final_epoch, final_best_path)

        self.save_training_history()

    def _create_warmup_cosine_scheduler(self):
        """Linear warmup then cosine decay to lr_min (multiplicative factors vs base_lr)."""

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            cosine_factor = (1 + np.cos(np.pi * progress)) / 2
            lr_ratio = self.lr_min / self.base_lr + (1 - self.lr_min / self.base_lr) * cosine_factor
            return lr_ratio

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_checkpoint(self, epoch: int, path: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'train_maes': self.train_maes,
            'val_losses': self.val_losses,
            'val_maes': self.val_maes,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def save_training_history(self):
        history = {
            'config': {
                'loss_type': self.loss_type,
                'learning_rate': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay'],
                'batch_size': self.config['batch_size'],
                'num_epochs': self.config['num_epochs'],
                'early_stopping_patience': self.early_stopping_patience,
            },
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'epochs': []
        }

        for epoch in range(len(self.train_losses)):
            history['epochs'].append({
                'epoch': epoch + 1,
                'train_loss': float(self.train_losses[epoch]),
                'train_mae': float(self.train_maes[epoch]),
                'val_loss': float(self.val_losses[epoch]),
                'val_mae': float(self.val_maes[epoch])
            })

        with open(self.history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.best_val_loss == float('inf') and 'best_val_mae' in checkpoint:
            self.best_val_loss = checkpoint['best_val_mae']
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        self.train_losses = checkpoint['train_losses']
        self.train_maes = checkpoint['train_maes']
        self.val_losses = checkpoint['val_losses']
        self.val_maes = checkpoint['val_maes']
        return checkpoint['epoch']
