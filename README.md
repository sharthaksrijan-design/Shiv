# Phase-SNN Phase 3

Phase-SNN is a Phase-based Spiking Neural Network architecture for intent classification and language modeling.

## Project Structure

- `phase_snn_v2.py`: Core encoder with upgrades (Balanced Quads, Sampled Metric Loss, Frequency Bands, K Partitioning).
- `phase_snn_v12.py`: Complex weights and sequence awareness.
- `phase_snn_torch.py`: PyTorch implementation with real weights.
- `checkpoint.py`: Utilities for saving and loading training states.
- `export_weights.py`: Exporting model weights to JSON for Rust inference.
- `baseline.py`: NumPy-based full pipeline for intent classification and generation PoC.
- `train.py`: PyTorch-based training script for Phase 3 Language Model.
- `rust/`: Rust implementation for high-performance inference.
- `requirements.txt`: Python dependencies.

## Setup Instructions (Kaggle)

To run this project on Kaggle:

1. **Create a new Kaggle Notebook.**
2. **Enable Internet:** Notebook Settings (right panel) → Internet → **ON**.
3. **Add GloVe dataset:** Click **+ Add Data** → search `danielwillgeorge/glove6b100dtxt` → Add.
4. **Upload the code:** You can either clone this repo into the Kaggle environment or copy-paste the scripts.
5. **Install dependencies:**
   ```python
   !pip install -r requirements.txt
   ```
6. **Run Baseline:**
   ```bash
   python baseline.py
   ```
7. **Run Training:**
   ```bash
   python train.py
   ```

## Guide to Running the Training

The `train.py` script automatically detects your environment (Kaggle, Colab, etc.) and configures checkpoint directories and hardware workers accordingly.

- **Checkpoints:** By default, it saves checkpoints to `/kaggle/working/phase_snn_ckpts` on Kaggle.
- **Hardware:** It will use the GPU if available.
- **PPL Progress:** Expect PPL to start high (~3000) and drop significantly towards ~40-80 after 30,000 steps.

## Future Optimizations

The following optimizations have been identified for future implementation:

1. **Remove Hardcoded Paths**: Currently, paths for Kaggle/Colab/Lightning.ai are hardcoded. These should be moved to a configuration file or environment variables.
2. **Modularize Data Loading**: Data loading logic in `baseline.py` and `train.py` can be abstracted into a common data utility module.
3. **Unified Optimizer**: The base64 embedded `Adam` class in `phase_snn_v2.py` and `phase_snn_v12.py` can be refactored into a single utility file.
4. **CI/CD**: Integrate GitHub Actions for automated testing and linting.
5. **Rust Bindings**: Create Python bindings for the Rust inference engine for easier integration.
6. **Configuration Management**: Use `hydra` or `argparse` more extensively for hyperparameter tuning.
