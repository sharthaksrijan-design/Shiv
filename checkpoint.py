"""
Checkpoint utilities for Phase-SNN training.
Saves and restores full training state so Colab timeouts
don't lose progress.

Usage:
    ckpt = CheckpointManager('/content/drive/MyDrive/phase_snn_ckpts',
                             prefix='lm_phase2')
    # In training loop:
    ckpt.save(step, model, optimizer, loss_hist, config)
    # On restart:
    state = ckpt.load_latest()
    if state:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_step = state['step'] + 1
        loss_hist  = state['loss_hist']
"""

import os, json, time
import torch
import numpy as np


class CheckpointManager:
    """
    Saves checkpoints to Google Drive (or any path).
    Keeps the last 3 checkpoints to save space.
    """

    def __init__(self, save_dir: str, prefix: str = 'phase_snn',
                 keep_last: int = 3):
        self.save_dir  = save_dir
        self.prefix    = prefix
        self.keep_last = keep_last
        os.makedirs(save_dir, exist_ok=True)

    def _path(self, step: int) -> str:
        return os.path.join(self.save_dir,
                            f'{self.prefix}_step{step:07d}.pt')

    def save(self, step: int,
             model: torch.nn.Module,
             optimizer: torch.optim.Optimizer,
             loss_hist: list,
             config: dict,
             extra: dict = None) -> str:
        """
        Save a checkpoint.
        Returns the path it was saved to.
        """
        t0 = time.time()
        state = {
            'step':       step,
            'model':      model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'loss_hist':  loss_hist[-1000:],   # keep last 1000 for plots
            'config':     config,
            'timestamp':  time.time(),
        }
        if extra:
            state.update(extra)

        path = self._path(step)
        # Save atomically: write to .tmp then rename
        tmp_path = path + '.tmp'
        torch.save(state, tmp_path)
        os.rename(tmp_path, path)

        elapsed = time.time() - t0
        size_mb = os.path.getsize(path) / 1e6
        print(f"  Checkpoint saved: step={step}  "
              f"size={size_mb:.1f}MB  t={elapsed:.1f}s  → {path}")

        # Clean up old checkpoints
        self._cleanup()
        return path

    def load_latest(self) -> dict | None:
        """
        Load the most recent checkpoint.
        Returns None if no checkpoints exist.
        """
        checkpoints = self._list_checkpoints()
        if not checkpoints:
            print("  No checkpoints found — starting from scratch")
            return None

        latest = checkpoints[-1]
        print(f"  Loading checkpoint: {latest}")
        state = torch.load(latest, map_location='cpu',
                           weights_only=False)
        print(f"  Resumed from step={state['step']}  "
              f"loss_hist={len(state['loss_hist'])} entries")
        return state

    def load_step(self, step: int) -> dict | None:
        """Load a specific step's checkpoint."""
        path = self._path(step)
        if not os.path.exists(path):
            print(f"  Checkpoint not found: {path}")
            return None
        return torch.load(path, map_location='cpu', weights_only=False)

    def _list_checkpoints(self) -> list:
        """Return sorted list of checkpoint paths."""
        files = [
            os.path.join(self.save_dir, f)
            for f in os.listdir(self.save_dir)
            if f.startswith(self.prefix) and f.endswith('.pt')
        ]
        return sorted(files)

    def _cleanup(self):
        """Remove old checkpoints, keep the last N."""
        checkpoints = self._list_checkpoints()
        for old in checkpoints[:-self.keep_last]:
            os.remove(old)
            print(f"  Removed old checkpoint: {old}")

    def list(self):
        """Print all available checkpoints."""
        ckpts = self._list_checkpoints()
        if not ckpts:
            print("  No checkpoints found")
        for path in ckpts:
            state = torch.load(path, map_location='cpu',
                               weights_only=False)
            size  = os.path.getsize(path) / 1e6
            ts    = time.strftime('%Y-%m-%d %H:%M',
                                  time.localtime(state['timestamp']))
            print(f"  step={state['step']:>7}  "
                  f"loss={state['loss_hist'][-1]:.4f}  "
                  f"size={size:.1f}MB  saved={ts}")


def mount_drive_if_needed(checkpoint_dir: str) -> str:
    """
    Mount Google Drive if the checkpoint dir starts with /drive.
    Returns the actual path to use.
    """
    if checkpoint_dir.startswith('/content/drive'):
        try:
            from google.colab import drive
            if not os.path.exists('/content/drive/MyDrive'):
                print("Mounting Google Drive...")
                drive.mount('/content/drive')
                print("Drive mounted ✓")
            else:
                print("Drive already mounted ✓")
        except ImportError:
            print("Not in Colab — using local path")
            checkpoint_dir = checkpoint_dir.replace(
                '/content/drive/MyDrive', '/tmp/phase_snn_ckpts')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir
