# Neural Network Optimization & Deployment Pipeline

This project was developed as part of a research workflow focused on neural network optimization for edge ML deployment. It integrates **training log analysis**, **model pruning with linear regression**, **quantization**, and **ONNX export** for deployment on resource-constrained devices such as Arduino and Edge Impulse-compatible platforms.

The pipeline is designed to aid **automated model comparison** across activation functions, compression techniques, and deployment targets.

---

## 🔍 Features

- 🔧 **Structural pruning** with weight rebalancing using linear regression
- 🧠 **Training with multiple activation functions** (ReLU, Tanh, Sigmoid, etc.)
- 📊 **Smart analysis** of training logs and pruning effects
- 📦 **ONNX export** and Edge Impulse integration for microcontroller deployment
- ⚙️ **Profiling and inference scripts** to compare original and pruned models
- 🛠️ **Deployment-ready inference** with micro-benchmarking support

---

## 🗂 Project Structure

configs/      - Configuration files for datasets and models
src/          - Core pipeline codebase
models/       - Neural network definitions
pruning/      - Pruning methods and regression modules
training/     - Training utilities
export/       - ONNX export and Edge Impulse compatibility
inference/    - Inference and comparison scripts
deployment/   - Edge deployment helpers (e.g., Arduino)
tagging/      - Data tagging and annotation
data/         - Data prep and augmentation
analysis/     - Performance metrics and visualization
utils/        - Utility functions
scripts/      - End-to-end pipeline runners
results/      - Output directories with logs, ONNX files, and metrics

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt

2. Configure

Edit or create a dataset-specific YAML config in configs/.

3. Run Pipeline

python scripts/pipeline/pipeline_export_onnx_for_ei.py --config configs/config_your_dataset.yaml

This will train, prune, quantize, export, and prepare the model for Edge Impulse deployment.

⸻

📈 Sample Results

Coming soon: Visualizations comparing original and pruned models across accuracy, latency, and model size.

⸻

🛠 Dependencies
	•	PyTorch
	•	ONNX, ONNX Runtime
	•	scikit-learn
	•	NumPy
	•	PyYAML

⸻

📄 License

MIT License — see the LICENSE file.

⸻

🧠 Acknowledgments

This pipeline builds upon foundational work in model pruning, neural architecture optimization, and deployment for edge AI systems. It integrates insights from recent pruning and dynamic sparsity research.

⸻

📘 Coming Soon
	•	A blog post explaining how this project connects RAG-based log analysis with deployment-aware neural optimization.
	•	Scripts for Arduino MKR1000 inference with benchmark logs.

---

Let me know when your repo is up — I can help you write the repo description, topics/tags, and a short email to Daniel linking it. Want that next?
