import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def compute_harmfulness_direction(harmful_acts, harmless_acts):
    """
    Computes the harmfulness direction as the difference between the mean 
    harmful activation and the mean harmless activation.
    
    Args:
        harmful_acts (torch.Tensor): Shape (N_harmful, hidden_dim)
        harmless_acts (torch.Tensor): Shape (N_harmless, hidden_dim)
        
    Returns:
        torch.Tensor: The harmfulness direction vector (hidden_dim,)
    """
    mean_harmful = torch.mean(harmful_acts, dim=0)
    mean_harmless = torch.mean(harmless_acts, dim=0)
    direction = mean_harmful - mean_harmless
    return direction

def compute_refusal_direction(model, tokenizer, instructions, device="cpu"):
    """
    Computes the refusal direction using contrastive suffixes.
    For each instruction, we append a refusal prefix and a compliance prefix.
    Direction = Mean(Act(Refusal) - Act(Compliance))
    
    Args:
        model: The LLM
        tokenizer: The tokenizer
        instructions: List of strings (prompts)
        
    Returns:
        dict: layer_idx -> refusal_direction vector
    """
    refusal_suffixes = ["I cannot", "I can't", "I am unable to", "Sorry, but I cannot"]
    compliance_suffixes = ["Sure", "Here is", "Certainly", "Okay"]
    
    # For simplicity, we'll pick one representative pair to start, or average over multiple.
    # The paper often uses "I cannot" vs "Sure, here is"
    
    refusal_suffix = " I cannot"
    compliance_suffix = " Sure"
    
    # We need to construct prompts. 
    # Important: The formatting depends on the model's chat template.
    # TinyLlama uses <|user|>\n ... </s>\n<|assistant|>\n
    # We want to steer the *start* of the assistant response.
    
    refusal_prompts = []
    compliance_prompts = []
    
    # We will use the chat template up to the assistant generation start
    
    chat_template_start = "<|user|>\n"
    chat_template_mid = "</s>\n<|assistant|>\n"
    
    # Actually, let's just use the raw tokenizer apply_chat_template if possible, 
    # but manually constructing it ensures we control the very last tokens.
    
    # TinyLlama Chat format:
    # <|user|>
    # {message} </s>
    # <|assistant|>
    
    for inst in instructions:
        # Prompt ending in the suffix
        # We assume strict template adherence
        base = f"<|user|>\n{inst} </s>\n<|assistant|>\n"
        
        refusal_prompts.append(base + refusal_suffix)
        compliance_prompts.append(base + compliance_suffix)
        
    # Now we need activations for the *last token* of these prompts.
    # This corresponds to the token of the suffix.
    
    from utils import get_activations_layer_wise
    
    # We process in batches if needed, but here all at once for simplicity (~100 prompts)
    refusal_acts_dict = get_activations_layer_wise(model, tokenizer, refusal_prompts, device)
    compliance_acts_dict = get_activations_layer_wise(model, tokenizer, compliance_prompts, device)
    
    directions = {}
    
    # Keys should match because they come from the same model forward passes structure
    for layer in refusal_acts_dict.keys():
        r_acts = refusal_acts_dict[layer] # (N, dim)
        c_acts = compliance_acts_dict[layer] # (N, dim)
        
        # Difference per pair, then mean
        diffs = r_acts - c_acts
        mean_diff = torch.mean(diffs, dim=0)
        directions[layer] = mean_diff
        
    return directions

def compute_cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two 1D tensors.
    """
    v1 = vec1.detach().cpu().numpy().reshape(1, -1)
    v2 = vec2.detach().cpu().numpy().reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]

def compute_projections(direction, activations):
    """
    Computes the projection of activations onto the direction vector.
    
    Args:
        direction (torch.Tensor): Shape (hidden_dim,)
        activations (torch.Tensor): Shape (N, hidden_dim)
        
    Returns:
        np.array: Shape (N,) of projection scalar values
    """
    # Normalize direction? 
    # Usually we project onto the *unit* vector of the direction to get the component.
    # But if we want to see alignment with the raw direction magnitude, we might iterate.
    # Let's normalize the direction for standard projection concept.
    
    direction = direction.to(activations.device)
    norm = torch.norm(direction)
    if norm > 1e-6:
        unit_dir = direction / norm
    else:
        unit_dir = direction
        
    projections = torch.matmul(activations, unit_dir)
    return projections.detach().cpu().numpy()

def compute_norms(directions_dict):
    """
    Computes the L2 norm of the direction vectors at each layer.
    """
    norms = []
    layers = sorted(directions_dict.keys())
    for l in layers:
        vec = directions_dict[l]
        norms.append(torch.norm(vec).item())
    return norms
