import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_head_contributions(model, tokenizer, directions_dict, prompts, layer_indices=None):
    """
    Computes the contribution of each attention head to the directions at each layer.
    Contribution = Dot(Head_Output, Direction) for the last token.
    
    Returns:
        dict: layer_idx -> shape (num_heads, num_prompts) contribution matrix
    """
    if layer_indices is None:
        layer_indices = sorted(directions_dict.keys())
        
    contributions = {}
    
    # We need to hook o_proj input
    # TinyLlama: model.model.layers[i].self_attn.o_proj
    
    # We'll process layer by layer to save memory if needed, or all at once?
    # Let's do one forward pass if we can hook everything.
    
    # Storage for hook data: layer -> (batch, seq, concat_head_dim)
    hook_data = {}
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            # Input to o_proj is (hidden_states,)
            # hidden_states: (batch, seq, num_heads * head_dim)
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            # Save ref to CPU to avoid GPU OOM if many prompts
            hook_data[layer_idx] = inp.detach().cpu()
        return hook
    
    handles = []
    # Check max valid layer index
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "layers"):
        num_layers = len(model.layers)
    else:
        num_layers = model.config.num_hidden_layers
        
    valid_indices = [l for l in layer_indices if l < num_layers]
    
    for l in valid_indices:
        if hasattr(model, "model"):
            layer_module = model.model.layers[l].self_attn.o_proj
        else:
            # Fallback for some architectures might differ
            layer_module = model.layers[l].self_attn.o_proj
            
        h = layer_module.register_forward_hook(get_hook(l))
        handles.append(h)
        
    # Run forward pass
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
    
    with torch.no_grad():
        _ = model(**inputs)
        
    # Remove hooks
    for h in handles:
        h.remove()
        
    # Compute contributions
    # We need model config for num_heads and head_dim
    config = model.config
    num_heads = config.num_attention_heads
    hidden_dim = config.hidden_size
    head_dim = hidden_dim // num_heads
    
    # Indices of last tokens
    last_token_idx = inputs['attention_mask'].sum(dim=1).cpu() - 1
    
    for l in layer_indices:
        if l not in hook_data: continue
        
        # (batch, seq, hidden_dim) -> (batch, hidden_dim) at last token
        layer_inp = hook_data[l] #(B, S, H)
        
        # Extract last token vector
        # This input is CONCATENATED HEAD OUTPUTS (before mixing weights)
        # Actually o_proj mixes them. We need to unmix.
        # Wait, o_proj input IS the concatenated head outputs.
        # So Input[..., i*head_dim : (i+1)*head_dim] is the output of head i.
        
        # Now we need the corresponding weights from o_proj to project them to residual stream.
        # o_proj.weight: (hidden_dim, hidden_dim)
        # Slice weights for head i: weights[:, i*head_dim : (i+1)*head_dim]
        
        if hasattr(model, "model"):
            weight = model.model.layers[l].self_attn.o_proj.weight.detach().cpu() #(Out, In)
        else:
            weight = model.layers[l].self_attn.o_proj.weight.detach().cpu()
            
        direction = directions_dict[l].detach().cpu() #(hidden_dim,)
        
        # We want: Dot( W @ h_i, dir ) = h_i @ W.T @ dir
        # Let projected_dir = W.T @ dir  --> Shape (hidden_dim,)
        # Then Contribution = Dot(h_i, projected_dir_slice)
        
        projected_dir = torch.matmul(weight.T, direction) #(hidden_dim,)
        
        heads_contribs = []
        
        for i in range(num_heads):
            # Slice range
            start = i * head_dim
            end = (i+1) * head_dim
            
            # Get projected direction slice for this head
            dir_slice = projected_dir[start:end] #(head_dim,)
            
            # Get head outputs at last token
            # layer_inp: (B, S, H)
            head_outs = []
            for b in range(layer_inp.shape[0]):
                idx = last_token_idx[b]
                vec = layer_inp[b, idx, start:end] #(head_dim,)
                head_outs.append(vec)
            head_outs = torch.stack(head_outs) #(B, head_dim)
            
            # Dot product
            contrib = torch.matmul(head_outs, dir_slice) #(B,)
            
            # Mean over batch?
            heads_contribs.append(contrib.mean().item())
            
        contributions[l] = np.array(heads_contribs)
        
    return contributions

def plot_head_heatmap(contributions_dict, title, filename):
    """
    Plots a heatmap of Layer x Head contributions.
    """
    layers = sorted(contributions_dict.keys())
    if not layers: return
    
    num_heads = len(contributions_dict[layers[0]])
    
    matrix = np.zeros((len(layers), num_heads))
    
    for i, l in enumerate(layers):
        matrix[i, :] = contributions_dict[l]
        
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap="RdBu_r", center=0)
    plt.title(title)
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    plt.savefig(filename)
    plt.close()
    print(f"Saved heatmap to {filename}")
    
    # Return top heads (layer, head_idx)
    flat_indices = np.argsort(matrix.flatten())
    # Top 3 positive and Top 3 negative? Or just absolute magnitude?
    # Usually "Harmfulness Head" contributes POSITIVELY to the direction.
    
    top_indices = flat_indices[-5:][::-1] # indices of top 5
    
    top_heads = []
    for idx in top_indices:
        l_idx = idx // num_heads
        h_idx = idx % num_heads
        real_l = layers[l_idx]
        top_heads.append((real_l, h_idx))
        
    return top_heads

def visualize_attention_patterms(model, tokenizer, head_list, prompt, filename_prefix="attn"):
    """
    Visualizes attention matrix for specific heads.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    attentions = outputs.attentions # tuple of (batch, num_heads, seq, seq)
    
    tokens = [tokenizer.decode(t).replace(" ", "") for t in inputs['input_ids'][0]]
    
    for (layer, head) in head_list:
        attn_mat = attentions[layer][0, head, :, :].detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_mat, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
        plt.title(f"Attention Pattern L{layer} H{head}")
        plt.savefig(f"{filename_prefix}_L{layer}_H{head}.png")
        plt.close()

def visualize_aggregated_attention(model, tokenizer, head_tuple, prompts, filename):
    """
    Visualizes attention maps for a specific head across multiple prompts in a grid.
    head_tuple: (layer, head)
    prompts: list of strings
    """
    layer, head = head_tuple
    
    num_prompts = len(prompts)
    # Grid size: 1 row, N columns (if small N), or wrap
    cols = min(num_prompts, 5)
    rows = (num_prompts + cols - 1) // cols
    
    plt.figure(figsize=(4 * cols, 4 * rows))
    
    for i, p in enumerate(prompts):
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        # attentions: tuple of (batch, num_heads, seq, seq)
        # get matrix
        attn_mat = outputs.attentions[layer][0, head, :, :].detach().cpu().numpy()
        tokens = [tokenizer.decode(t).replace(" ", "") for t in inputs['input_ids'][0]]
        
        ax = plt.subplot(rows, cols, i+1)
        sns.heatmap(attn_mat, xticklabels=tokens, yticklabels=tokens, cmap="viridis", cbar=False, ax=ax)
        ax.set_title(f"Sample {i+1}")
        
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved aggregated attention plot to {filename}")
