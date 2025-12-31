"""
Standalone script to run layer-wise ablation study.
"""
import torch
from utils import load_model, get_data, get_activations_layer_wise
from probing import compute_harmfulness_direction, compute_refusal_direction
from ablation import run_ablation_sweep, plot_ablation_effects

# Chat template tokens
USER_START = chr(60) + "|user|" + chr(62) + chr(10)  # 
