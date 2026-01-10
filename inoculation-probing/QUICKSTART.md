# Quick Start Guide

## 🎉 Repository Setup Complete!

Your inoculation probing experiment framework is ready to use. Here's what's been created:

### 📁 Project Structure

```
inoculation-probing/
├── src/                          # Source code
│   ├── data/                     # Dataset builder
│   ├── extraction/               # Activation extractor
│   ├── probing/                  # Probe experiments
│   ├── prompts/                  # Prompt templates
│   └── utils/                    # Utilities (I/O, logging, stats)
├── experiments/                  # Experiment scripts
│   └── run_mvp_experiment.py     # Main MVP script
├── notebooks/                    # Jupyter notebooks
│   └── mvp_experiment.ipynb      # Colab-ready notebook
├── README.md                     # Full documentation
├── requirements.txt              # Dependencies
└── setup.py                      # Package setup
```

## 🚀 Next Steps

### 1. Push to GitHub

Create a new repository on GitHub and push:

```bash
# Create a new repository on GitHub (e.g., "inoculation-probing")
# Then run:

cd /home/user/Probing-experiments/inoculation-probing
git remote add origin https://github.com/YOUR-USERNAME/inoculation-probing.git
git push -u origin main
```

### 2. Set Up Colab

**Option A: Clone to Google Drive**

1. Open Google Drive
2. Create folder: `inoculation-probing`
3. Upload all files OR clone from GitHub:

```bash
# In Colab:
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR-USERNAME/inoculation-probing.git /content/drive/MyDrive/inoculation-probing
```

**Option B: Use Colab directly with GitHub**

1. Open the notebook from GitHub: `notebooks/mvp_experiment.ipynb`
2. Click "Open in Colab"
3. Mount Google Drive for saving results

### 3. Authenticate with HuggingFace

Llama 3.2-1B is a gated model. Get access:

1. Go to: https://huggingface.co/meta-llama/Llama-3.2-1B
2. Request access (usually instant)
3. Create HF token: https://huggingface.co/settings/tokens
4. In Colab:

```python
from huggingface_hub import notebook_login
notebook_login()  # Enter your token
```

### 4. Run the MVP Experiment

**Option A: Using the Colab Notebook (Recommended)**

1. Open `notebooks/mvp_experiment.ipynb` in Colab
2. Run all cells
3. Results will be saved to your Google Drive

**Option B: Using the Python Script**

```bash
cd /home/user/Probing-experiments/inoculation-probing
python experiments/run_mvp_experiment.py
```

## 🎯 What the Experiment Does

1. **Creates Dataset**: Spuriously correlated sentiment data
   - Train: Short→Positive, Long→Negative
   - Test: Anti-correlated

2. **Tests Prompts**: Compares baseline vs inoculation prompts
   - Baseline: No prompt
   - Inoculation: "Ignore length, focus on sentiment"
   - Anti-inoculation: "Pay attention to length" (should worsen)

3. **Evaluates Probes**: Trains linear probes on activations
   - Overall accuracy
   - Worst-group accuracy (key metric!)
   - Bootstrap confidence intervals

4. **Makes Decision**: GO/NO-GO for further investigation
   - GO (>10% improvement): Proceed with mechanistic analysis
   - NO-GO (<5% improvement): Document negative results

## 📊 Expected Output

The experiment will create:

```
results/
├── datasets/
│   └── mvp_dataset/              # Cached dataset
├── activations/                  # Cached activations (speeds up reruns)
├── experiments/
│   └── mvp_experiment/
│       ├── viz_*.png             # Visualizations
│       ├── probes/               # Trained probe models
│       ├── *.json                # Experiment logs
│       └── results_summary.csv   # Summary table
```

## ⚙️ Customization

### Change Dataset Size

```python
# In notebook or script
CONFIG = {
    'n_train_per_group': 200,  # Increase from 100
    'n_test_per_group': 100,   # Increase from 50
    # ...
}
```

### Test Different Layers

```python
CONFIG = {
    'layers': [8, 12, 16, 20, 24],  # Test more layers
    # ...
}
```

### Add Custom Prompts

Edit `src/prompts/templates.py`:

```python
PROMPT_TEMPLATES['my_custom_prompt'] = "Your custom prompt here.\n\n{text}"
```

Then add to experiment:

```python
PROMPTS = ['baseline', 'my_custom_prompt', 'anti_inoc_v1']
```

## 🐛 Troubleshooting

### "No module named 'src'"

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/inoculation-probing"
```

Or in Python:

```python
import sys
sys.path.insert(0, '/path/to/inoculation-probing')
```

### "Model not found" or "Access denied"

Make sure you:
1. Requested access to Llama 3.2-1B on HuggingFace
2. Authenticated with `huggingface-cli login` or `notebook_login()`

### "CUDA out of memory"

Reduce batch size in extraction:

```python
extractor.extract(texts, batch_size=4)  # Default is 8
```

Or use CPU:

```python
extractor = ActivationExtractor(
    model_name='meta-llama/Llama-3.2-1B',
    device='cpu'  # Use CPU instead of GPU
)
```

### Slow extraction

Enable caching (default):

```python
extractor = ActivationExtractor(
    cache_dir='./results/activations',  # Saves activations
    use_cache=True
)
```

Subsequent runs with same prompts will load from cache!

## 📚 Documentation

- **Full README**: `README.md`
- **Code Documentation**: Docstrings in all modules
- **Experiment Script**: `experiments/run_mvp_experiment.py`
- **Notebook**: `notebooks/mvp_experiment.ipynb`

## 🤝 Need Help?

Check the README for detailed documentation, or open an issue on GitHub.

---

**Ready to start probing! 🔍**
