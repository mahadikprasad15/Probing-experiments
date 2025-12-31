import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from utils import get_activations_layer_wise

def generate_4way_data(harmful_prompts, harmless_prompts):
    """
    Generates 4 conditions:
    1. Harmful + Refusal (HR): "I cannot"
    2. Harmful + Compliance (HC): "Sure"
    3. Harmless + Refusal (BR): "I cannot"
    4. Harmless + Compliance (BC): "Sure"
    
    Returns list of dicts with 'text', 'type', 'label'
    """
    data = []
    
    # TinyLlama format: <|user|>\n{Instruction} </s>\n<|assistant|>\n{Response}
    
    refusal_suffix = " I cannot"
    compliance_suffix = " Sure"
    
    for p in harmful_prompts:
        base = f"<|user|>\n{p} </s>\n<|assistant|>\n"
        data.append({"text": base + refusal_suffix, "type": "Harmful", "response": "Refusal", "label": "HR"})
        data.append({"text": base + compliance_suffix, "type": "Harmful", "response": "Compliance", "label": "HC"})
        
    for p in harmless_prompts:
        base = f"<|user|>\n{p} </s>\n<|assistant|>\n"
        data.append({"text": base + refusal_suffix, "type": "Harmless", "response": "Refusal", "label": "BR"})
        data.append({"text": base + compliance_suffix, "type": "Harmless", "response": "Compliance", "label": "BC"})
        
    return data

def extract_specific_activations(model, tokenizer, data_list, layer_indices):
    """
    Extracts activations at:
    1. End of Instruction (Pos 1) - Before response generation
    2. End of Response Prefix (Pos 2) - After "I cannot"/"Sure"
    """
    # We need to process carefully to get the exact token indices.
    # To simplify, we can tokenize and find the position of the last token (Pos 2)
    # And finding Pos 1 is harder without knowing the prompt length.
    
    results = {l: {"pos1": [], "pos2": []} for l in layer_indices}
    labels = [d["label"] for d in data_list]
    types = [d["type"] for d in data_list]
    responses = [d["response"] for d in data_list]
    
    # Process one by one? Or batch? One by one is safer for index logic.
    
    for item in data_list:
        text = item["text"]
        
        # Tokenize
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        input_ids = enc.input_ids
        
        # Identify positions
        # The prompt ends with "Sure" or "I cannot"
        # "Sure" is usually 1 token. "I cannot" is 2 tokens.
        # But wait, looking at: "<|assistant|>\n"
        # We want the token BEFORE the response starts? OR the last token of instruction?
        # Detailed paper methodology: 
        # "Pos 1": Last token of user instruction.
        # "Pos 2": Last token of the forced prefix.
        
        # Let's find the index of "<|assistant|>"
        # TinyLlama tokenizer might tokenize this as special tokens or raw.
        # Let's assume we can find the boundary by length of the "base" part.
        
        # Re-construct base to find length
        # But data_list item doesn't have raw prompt separated easily if we just used strings.
        # We know the suffixes.
        
        if text.endswith(" Sure"):
            suffix_len_chars = len(" Sure")
        elif text.endswith(" I cannot"):
            suffix_len_chars = len(" I cannot")
        else:
            raise ValueError(f"Unknown suffix in {text}")
            
        base_text = text[:-suffix_len_chars]
        
        # Tokenize base
        base_enc = tokenizer(base_text, return_tensors="pt")
        pos1_idx = base_enc.input_ids.shape[1] - 1  # Last token of base (includes <|assistant|> header usually?)
        # Actually base includes "\n<|assistant|>\n"
        # Ideally Pos 1 is the last token of the INSTRUCTION, i.e. BEFORE </s>?
        # Paper says "Last token of the user instruction".
        # <|user|>\n{Instruction} </s> ...
        # So we should find </s> index?
        
        # Let's look for tokenizer.eos_token_id (</s>)
        eos_indices = (input_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            pos1_idx = eos_indices[0].item() # The </s> token. Or the one before?
            # Let's use the </s> token as the anchor for "End of Instruction"
        else:
            # Fallback
            pos1_idx = base_enc.input_ids.shape[1] - 5 # Approximate
            
        pos2_idx = input_ids.shape[1] - 1 # Last token of ' I cannot' or ' Sure'
        
        # Forward pass output_hidden_states
        with torch.no_grad():
            outputs = model(**enc, output_hidden_states=True)
            
        for l in layer_indices:
            # hidden_states tuple
            # index l (accounts for embedding layer at 0?)
            # Usually outputs.hidden_states[l] if we access directly relative to config layers?
            # outputs.hidden_states has len(layers) + 1.
            # layer_indices usually 0..21. +1 for embedding.
            # Standard: layer i output is hidden_states[i+1]
            
            h = outputs.hidden_states[l+1] #(1, seq, dim)
            
            vec1 = h[0, pos1_idx, :].cpu().numpy()
            vec2 = h[0, pos2_idx, :].cpu().numpy()
            
            results[l]["pos1"].append(vec1)
            results[l]["pos2"].append(vec2)
            
    return results, labels, types, responses

def plot_pca_clustering(results, labels, types, responses, layer_idx, filename_prefix="pca"):
    """
    Plots PCA at Pos 1 and Pos 2 for a specific layer.
    """
    if layer_idx not in results:
        print(f"Layer {layer_idx} not in results.")
        return
        
    data = results[layer_idx]
    
    # Prepare Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    positions = ["pos1", "pos2"]
    titles = ["Pos 1: End of Instruction", "Pos 2: End of Response Prefix"]
    
    for i, pos in enumerate(positions):
        X = np.array(data[pos]) #(N, dim)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Plot
        ax = axes[i]
        
        # Markers: Refusal (X) vs Compliance (O)
        # Colors: Harmful (Red) vs Harmless (Blue)
        
        # Iterate points
        for j in range(len(X)):
            x, y = X_pca[j]
            t = types[j] # Harmful/Harmless
            r = responses[j] # Refusal/Compliance
            
            color = "red" if t == "Harmful" else "blue"
            marker = "x" if r == "Refusal" else "o"
            
            ax.scatter(x, y, c=color, marker=marker, s=100, alpha=0.7)
            
        ax.set_title(f"{titles[i]} (Layer {layer_idx})")
        
        # Creating custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Compliance', markerfacecolor='gray', markersize=10),
            Line2D([0], [0], marker='x', color='w', label='Refusal', markeredgecolor='gray', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Harmful', markerfacecolor='red', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Harmless', markerfacecolor='blue', markersize=10)
        ]
        ax.legend(handles=legend_elements)
        
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_L{layer_idx}.png")
    plt.close()
    print(f"Saved PCA plot to {filename_prefix}_L{layer_idx}.png")
