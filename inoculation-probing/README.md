# Inoculation Probing: Can Prompts Disentangle Spurious Correlations?

**Experimental framework for testing whether inoculation prompts can improve probe generalization in the presence of spurious correlations.**

## 📋 Overview

This repository implements an experiment to test the hypothesis:

> **"Inoculation prompts can disentangle spurious correlations in frozen language models, making linear probes more robust to distribution shifts."**

Inspired by the [Inoculation Prompting paper](https://arxiv.org/abs/2410.02135), we test whether prompts alone (without fine-tuning) can reduce spurious feature reliance in model representations.

## 🧪 Experiment Design

### The Setup

1. **Spurious Dataset**: Create a sentiment dataset with a spurious length correlation
   - Train: Short reviews → Positive, Long reviews → Negative
   - Test: **Anti-correlated** (Short → Negative, Long → Positive)

2. **Baseline**: Train a probe on a frozen model
   - Expected: Low accuracy (probe learns "length" instead of "sentiment")

3. **Inoculation**: Apply prompts that redirect attention away from length
   - Examples: "Ignore the length and focus on sentiment"
   - Expected: Improved accuracy if prompts disentangle features

4. **Mechanistic Analysis**: If successful, investigate *how* prompts work
   - Attention pattern analysis
   - Layer-wise probing
   - Activation geometry

### Key Question

Does an inoculation prompt act as a **filter** (disentangles features) or a **highlighter** (amplifies spurious correlations)?

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Open the notebook: [`notebooks/mvp_experiment.ipynb`](notebooks/mvp_experiment.ipynb)
2. Mount your Google Drive
3. Authenticate with HuggingFace (for Llama 3.2-1B access)
4. Run all cells

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/mahadikprasad15/inoculation-probing.git
cd inoculation-probing

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace (required for Llama 3.2-1B)
huggingface-cli login

# Run MVP experiment
python experiments/run_mvp_experiment.py
```

## 📁 Project Structure

```
inoculation-probing/
├── src/
│   ├── data/                  # Dataset creation
│   │   └── dataset_builder.py # Spurious correlation dataset builder
│   ├── extraction/            # Activation extraction
│   │   └── activations.py     # Extract from LLM layers
│   ├── probing/               # Probe training
│   │   └── probe.py           # Linear probe experiments
│   ├── prompts/               # Prompt templates
│   │   └── templates.py       # Inoculation prompts
│   └── utils/                 # Utilities
│       ├── io.py              # Save/load helpers
│       ├── logging.py         # Experiment tracking
│       └── stats.py           # Bootstrap CI, metrics
├── experiments/               # Experiment scripts
│   └── run_mvp_experiment.py  # Main MVP script
├── notebooks/                 # Jupyter notebooks
│   └── mvp_experiment.ipynb   # Colab-ready notebook
├── results/                   # Saved results (created at runtime)
│   ├── datasets/              # Cached datasets
│   ├── activations/           # Cached activations
│   └── experiments/           # Experiment logs & artifacts
└── requirements.txt
```

## 🔧 Usage

### 1. Create Spuriously Correlated Dataset

```python
from src.data import SpuriousDatasetBuilder

dataset = SpuriousDatasetBuilder(
    source_dataset='imdb',
    n_train_per_group=100,     # 100 short+pos, 100 long+neg
    n_test_per_group=50,       # 50 short+neg, 50 long+pos
    short_range=(10, 30),      # Token count for "short"
    long_range=(100, 200),     # Token count for "long"
    seed=42
)

splits = dataset.create_splits()
dataset.save('./results/datasets/my_dataset')
```

### 2. Extract Activations

```python
from src.extraction import ActivationExtractor
from src.prompts import format_prompt

# Initialize extractor
extractor = ActivationExtractor(
    model_name='meta-llama/Llama-3.2-1B',
    layer=12,  # Middle layer
    cache_dir='./results/activations'  # Enable caching
)

# Extract with different prompts
baseline_acts = extractor.extract(
    texts=['Good movie!', 'Terrible film.'],
    prompt_name='baseline'
)

inoculated_acts = extractor.extract(
    texts=[format_prompt('inoc_imperative_v2', t) for t in texts],
    prompt_name='inoc_imperative_v2'
)
```

### 3. Train and Evaluate Probe

```python
from src.probing import ProbeExperiment

probe_exp = ProbeExperiment(
    activations_train=train_acts,
    labels_train=train_labels,
    activations_test=test_acts,
    labels_test=test_labels,
    metadata_test={
        'length_labels': test_lengths,
        'sentiment_labels': test_sentiments
    }
)

# Train with cross-validated regularization
probe_exp.train_probe(regularization='cv')

# Evaluate with bootstrap CI
metrics = probe_exp.evaluate(n_bootstrap=100)

# Visualize
probe_exp.visualize_results(save_path='./results/probe_viz.png')
```

### 4. Run Full MVP Experiment

```python
from experiments.run_mvp_experiment import run_mvp_experiment

results = run_mvp_experiment(
    n_train_per_group=100,
    n_test_per_group=50,
    model_name='meta-llama/Llama-3.2-1B',
    layers=[12, 20],  # Test multiple layers
    prompts=None,     # Use MVP defaults
    base_dir='./results',
    use_cache=True
)
```

## 📊 Prompt Templates

The repository includes several prompt categories:

| Category | Example | Purpose |
|----------|---------|---------|
| **Baseline** | `{text}` | No intervention |
| **Imperative** | `"Ignore the length and focus on sentiment.\n\n{text}"` | Direct instruction |
| **Descriptive** | `"Review length is not indicative of sentiment.\n\n{text}"` | State relationship |
| **Role-playing** | `"You are an unbiased analyzer...\n\n{text}"` | Persona adoption |
| **Anti-inoculation** | `"Pay attention to the length.\n\n{text}"` | Should worsen performance |

See [`src/prompts/templates.py`](src/prompts/templates.py) for all templates.

## 🎯 Expected Outcomes

### Success (Prompts Disentangle)
- Inoculation prompts improve worst-group accuracy by >10%
- Anti-inoculation prompts decrease accuracy
- **Interpretation**: Prompts act as "filters" that redirect attention

### Failure (Prompts Don't Help)
- All prompts perform similarly to baseline
- **Interpretation**: Spurious correlations are "locked in" frozen representations
- **Learning**: Highlights limits of prompting vs. fine-tuning

### Mixed Results
- Some prompts help, others don't
- Layer-dependent effects
- **Interpretation**: Nuanced mechanistic story about when prompts work

## 🔬 Mechanistic Analysis (If Prompts Work)

If inoculation shows a strong effect, follow up with:

1. **Attention Analysis**: Where does attention shift?
   ```python
   # Compare attention patterns
   attn_baseline = get_attention(model, text)
   attn_inoculated = get_attention(model, inoculated_text)
   ```

2. **Layer-wise Probing**: At which layers does disentanglement occur?
   ```python
   for layer in range(0, 32, 2):
       probe_at_layer(layer, prompt_name)
   ```

3. **Activation Patching**: Which components are causal?
   ```python
   # Patch specific attention heads
   patch_head(layer=15, head=7, source='inoculated')
   ```

4. **Feature Direction Analysis**: How does activation geometry change?
   ```python
   # Measure angle between length and sentiment directions
   cos_sim(length_dir, sentiment_dir)  # Should decrease with inoculation
   ```

## 🔑 Google Colab Setup

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Set working directory
import os
os.chdir('/content/drive/MyDrive/inoculation-probing')
```

### 2. Install Dependencies

```python
!pip install -q -r requirements.txt
```

### 3. HuggingFace Authentication

```python
from huggingface_hub import notebook_login
notebook_login()  # Enter your HF token when prompted
```

### 4. Run Experiment

```python
from experiments.run_mvp_experiment import run_mvp_experiment

results = run_mvp_experiment(
    base_dir='/content/drive/MyDrive/inoculation-probing/results'
)
```

## 📈 Metrics

The framework computes:

- **Overall Accuracy**: Standard test accuracy
- **Worst-Group Accuracy**: Min accuracy across length×sentiment groups
- **Bootstrap CI**: 95% confidence intervals (1000 samples)
- **Stratified Metrics**: Per-group breakdown
  - Short+Positive
  - Short+Negative
  - Long+Positive
  - Long+Negative

## 🤝 Contributing

This is a research experiment. If you find interesting results or want to extend the framework:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 Citation

If you use this code or find it helpful, please cite:

```bibtex
@misc{inoculation-probing-2025,
  author = {Your Name},
  title = {Inoculation Probing: Can Prompts Disentangle Spurious Correlations?},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/mahadikprasad15/inoculation-probing}
}
```

## 📚 References

- **Inoculation Prompting**: [Eliciting traits from LLMs during training can reduce trait expression at test-time](https://arxiv.org/abs/2410.02135)
- **Linear Probing**: Understanding representations through linear classifiers
- **Spurious Correlations**: [Right for the Wrong Reasons](https://arxiv.org/abs/1703.03717)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🐛 Issues

Report issues at: https://github.com/mahadikprasad15/inoculation-probing/issues

---

**Happy probing! 🔍**
