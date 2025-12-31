"""
Head-level ablation study.
Ablates specific attention heads and measures effect on Harmfulness/Refusal projections.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class HeadAblator:
    """
    Ablates (zeros out) a specific attention head's output.
    """
    def __init__(self, model, layer_idx, head_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.hook_handle = None
        
        # Get head dimension
        if hasattr(model, "model"):
            self.num_heads = model.config.num_attention_heads
            self.head_dim = model.config.hidden_size // self.num_heads
        else:
            self.num_heads = model.config.num_attention_heads
            self.head_dim = model.config.hidden_size // self.num_heads
        
    def ablation_hook(self, module, input):
        """
        Pre-hook on the attention output projection (o_proj).
        We zero out the contribution from a specific head.
        
        The input to o_proj is the concatenated head outputs: (batch, seq, num_heads * head_dim)
        We zero out the slice corresponding to our target head.
        """
        # input is a tuple, input[0] is the actual tensor
        x = input[0]  # (batch, seq, num_heads * head_dim)
        
        # Calculate the slice for this head
        start_idx = self.head_idx * self.head_dim
        end_idx = start_idx + self.head_dim
        
        # Zero out this head's contribution
        x_modified = x.clone()
        x_modified[:, :, start_idx:end_idx] = 0
        
        # Return modified input (as tuple)
        return (x_modified,) + input[1:] if len(input) > 1 else (x_modified,)
    
    def attach(self):
        """Attach the ablation hook to the o_proj layer."""
        if hasattr(self.model, "model"):
            o_proj = self.model.model.layers[self.layer_idx].self_attn.o_proj
        else:
            o_proj = self.model.layers[self.layer_idx].self_attn.o_proj
            
        # Use register_forward_pre_hook to modify input to o_proj
        self.hook_handle = o_proj.register_forward_pre_hook(self.ablation_hook)
        
    def remove(self):
        """Remove the ablation hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None


def measure_projection(model, tokenizer, prompts, direction):
    """
    Measures the projection of activations onto a direction at the final layer.
    """
    direction = direction.to(model.device)
    direction_unit = direction / torch.norm(direction)
    
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    final_hidden = outputs.hidden_states[-1]
    last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
    
    projections = []
    for i in range(final_hidden.shape[0]):
        vec = final_hidden[i, last_token_idx[i], :]
        proj = torch.dot(vec, direction_unit).item()
        projections.append(proj)
        
    return np.mean(projections)


def run_head_ablation_study(model, tokenizer, prompts, directions_dict, heads_to_test, direction_name="Direction"):
    """
    Ablates each head in heads_to_test and measures effect.
    
    heads_to_test: list of (layer, head) tuples
    """
    # Get final layer direction
    if hasattr(model, "model"):
        num_layers = len(model.model.layers)
    else:
        num_layers = model.config.num_hidden_layers
        
    final_layer_idx = num_layers - 1
    if final_layer_idx not in directions_dict:
        final_layer_idx = max(directions_dict.keys())
    final_direction = directions_dict[final_layer_idx]
    
    # Baseline
    print(f"Measuring baseline for {direction_name}...")
    baseline = measure_projection(model, tokenizer, prompts, final_direction)
    print(f"Baseline: {baseline:.4f}")
    
    results = {}
    
    print(f"Testing {len(heads_to_test)} heads...")
    for layer, head in tqdm(heads_to_test):
        # Ensure valid indices
        if layer >= num_layers:
            continue
            
        ablator = HeadAblator(model, layer, head)
        ablator.attach()
        
        ablated = measure_projection(model, tokenizer, prompts, final_direction)
        
        ablator.remove()
        
        effect = baseline - ablated
        results[(layer, head)] = {
            'baseline': baseline,
            'ablated': ablated,
            'effect': effect,
            'relative_effect': effect / abs(baseline) * 100 if baseline != 0 else 0
        }
        
    return results, baseline


def plot_head_ablation_results(results, direction_name, filename):
    """
    Bar plot of head ablation effects.
    """
    heads = list(results.keys())
    effects = [results[h]['relative_effect'] for h in heads]
    labels = [f"L{l}H{h}" for l, h in heads]
    
    # Sort by absolute effect
    sorted_pairs = sorted(zip(labels, effects), key=lambda x: abs(x[1]), reverse=True)
    labels, effects = zip(*sorted_pairs)
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if e > 0 else 'blue' for e in effects]
    plt.bar(labels, effects, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Head (Layer, Head)')
    plt.ylabel('Relative Effect (%)')
    plt.title(f'Effect of Ablating Individual Heads on {direction_name} Projection')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")
    
    # Print top effects
    print(f"\nTop head effects for {direction_name}:")
    for label, eff in sorted_pairs[:5]:
        print(f"  {label}: {eff:.1f}%")
        
    return sorted_pairs
