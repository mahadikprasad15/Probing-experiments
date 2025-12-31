import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class DirectionAblator:
    """
    Ablates (projects out) a direction from the residual stream at a specific layer.
    """
    def __init__(self, model, layer_idx, direction):
        self.model = model
        self.layer_idx = layer_idx
        self.direction = direction.to(model.device)
        # Normalize direction
        self.direction_unit = self.direction / torch.norm(self.direction)
        self.hook_handle = None
        
    def ablation_hook(self, module, input, output):
        """
        Hook that projects out the direction from hidden states.
        output: tuple (hidden_states, ...) or just hidden_states
        hidden_states: (batch, seq, hidden_dim)
        """
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Project out the direction: h = h - (h · d_unit) * d_unit
        # Compute dot product: (batch, seq)
        proj = torch.matmul(hidden_states, self.direction_unit)
        # Subtract projection: (batch, seq, 1) * (hidden_dim,)
        ablated = hidden_states - proj.unsqueeze(-1) * self.direction_unit
        
        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        else:
            return ablated
    
    def attach(self):
        """Attach the ablation hook to the layer."""
        if hasattr(self.model, "model"):
            layer_module = self.model.model.layers[self.layer_idx]
        else:
            layer_module = self.model.layers[self.layer_idx]
        self.hook_handle = layer_module.register_forward_hook(self.ablation_hook)
        
    def remove(self):
        """Remove the ablation hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


def measure_final_projection(model, tokenizer, prompts, direction_at_final_layer):
    """
    Measures the projection of activations onto a direction at the final layer.
    Returns mean projection value across prompts.
    """
    direction = direction_at_final_layer.to(model.device)
    direction_unit = direction / torch.norm(direction)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get final layer hidden states
    final_hidden = outputs.hidden_states[-1]  # (batch, seq, dim)
    
    # Get last token position for each sequence
    last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
    
    projections = []
    for i in range(final_hidden.shape[0]):
        vec = final_hidden[i, last_token_idx[i], :]
        proj = torch.dot(vec, direction_unit).item()
        projections.append(proj)
        
    return np.mean(projections)


def run_ablation_sweep(model, tokenizer, prompts, directions_dict, direction_name="Direction"):
    """
    For each layer, ablate the direction at that layer and measure the 
    effect on the final layer's projection.
    
    Returns dict: layer_idx -> effect (baseline - ablated)
    """
    # Get number of layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    else:
        num_layers = model.config.num_hidden_layers
        
    # Final layer direction for measurement
    final_layer_idx = num_layers - 1
    if final_layer_idx not in directions_dict:
        # Use the highest available layer
        final_layer_idx = max(directions_dict.keys())
    final_direction = directions_dict[final_layer_idx]
    
    # Baseline: no ablation
    print(f"Measuring baseline projection for {direction_name}...")
    baseline = measure_final_projection(model, tokenizer, prompts, final_direction)
    print(f"Baseline projection: {baseline:.4f}")
    
    effects = {}
    
    # For each layer where we have a direction
    valid_layers = sorted([l for l in directions_dict.keys() if l < num_layers])
    
    print(f"Running ablation sweep across {len(valid_layers)} layers...")
    for layer_idx in tqdm(valid_layers):
        direction = directions_dict[layer_idx]
        
        # Attach ablation hook
        ablator = DirectionAblator(model, layer_idx, direction)
        ablator.attach()
        
        # Measure with ablation
        ablated_proj = measure_final_projection(model, tokenizer, prompts, final_direction)
        
        # Remove hook
        ablator.remove()
        
        # Effect = how much did ablation reduce the final projection?
        effect = baseline - ablated_proj
        effects[layer_idx] = {
            'baseline': baseline,
            'ablated': ablated_proj,
            'effect': effect,
            'relative_effect': effect / abs(baseline) if baseline != 0 else 0
        }
        
    return effects, baseline


def plot_ablation_effects(effects, direction_name, filename):
    """
    Plots the effect of ablating each layer.
    """
    layers = sorted(effects.keys())
    effect_values = [effects[l]['effect'] for l in layers]
    relative_effects = [effects[l]['relative_effect'] * 100 for l in layers]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Absolute effect
    ax1 = axes[0]
    ax1.bar(layers, effect_values, color='steelblue')
    ax1.set_xlabel('Layer Ablated')
    ax1.set_ylabel('Effect (Baseline - Ablated)')
    ax1.set_title(f'Effect of Ablating {direction_name} Direction')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Relative effect (%)
    ax2 = axes[1]
    ax2.bar(layers, relative_effects, color='darkorange')
    ax2.set_xlabel('Layer Ablated')
    ax2.set_ylabel('Relative Effect (%)')
    ax2.set_title(f'Relative Effect of Ablating {direction_name} Direction')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved ablation plot to {filename}")
    
    # Find most important layers
    sorted_by_effect = sorted(effects.items(), key=lambda x: abs(x[1]['effect']), reverse=True)
    print(f"\nTop 5 most causally important layers for {direction_name}:")
    for layer, data in sorted_by_effect[:5]:
        print(f"  Layer {layer}: Effect = {data['effect']:.4f} ({data['relative_effect']*100:.1f}%)")
        
    return sorted_by_effect
