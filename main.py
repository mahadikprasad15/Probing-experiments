import torch
import matplotlib.pyplot as plt
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import load_model, get_data, get_activations_layer_wise
from probing import compute_harmfulness_direction, compute_refusal_direction, compute_cosine_similarity
from cache_utils import cache_activations, cache_directions, cache_exists, load_from_cache, save_to_cache

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def output_path(filename):
    """Returns path in outputs directory."""
    return os.path.join(OUTPUT_DIR, filename)

def main():
    print("Starting replication experiment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    model, tokenizer = load_model()
    # model.to(device) # handled by device_map="auto"
    
    # 2. Get Data
    harmful_prompts, harmless_prompts = get_data()
    print(f"Loaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts.")
    
    # Adjust prompts for chat template for hidden state extraction
    # For harmfulness direction, we want the internal representation of the *instruction itself* 
    # or the state *right before* generation starts.
    # Usually: <|user|>\n{Instruction} </s>\n<|assistant|>\n
    # We want the act at the last token of this prompt.
    
    def format_prompts(prompts):
        return [f"<|user|>\n{p} </s>\n<|assistant|>\n" for p in prompts]
        
    formatted_harmful = format_prompts(harmful_prompts)
    formatted_harmless = format_prompts(harmless_prompts)
    
    # 3. Extract Activations (Layer-wise)
    # 3. Extract Activations (Layer-wise)
    print("Extracting activations for harmful prompts...")
    harmful_acts_dict = cache_activations(model, tokenizer, formatted_harmful, "harmful_acts", get_activations_layer_wise)
    
    print("Extracting activations for harmless prompts...")
    harmless_acts_dict = cache_activations(model, tokenizer, formatted_harmless, "harmless_acts", get_activations_layer_wise)
    
    # 4. Compute Harmfulness Direction (Layer-wise)
    print("Computing harmfulness directions...")
    harmfulness_directions = {}
    layers = sorted(harmful_acts_dict.keys())
    
    for layer in layers:
        h_acts = harmful_acts_dict[layer]
        b_acts = harmless_acts_dict[layer]
        # Ensure sizes match if needed (we have 20 vs 20 so ok)
        harmfulness_directions[layer] = compute_harmfulness_direction(h_acts, b_acts)
        
    # 5. Compute Refusal Direction (Layer-wise)
    # 5. Compute Refusal Direction (Layer-wise)
    print("Computing refusal directions using contrastive suffixes...")
    # we cache this because it runs model inference
    if cache_exists("refusal_directions"):
        refusal_directions = load_from_cache("refusal_directions")
    else:
        refusal_directions = compute_refusal_direction(model, tokenizer, harmful_prompts, device)
        save_to_cache(refusal_directions, "refusal_directions")
    
    # 6. Analysis: Cosine Similarity over layers
    similarities = []
    layer_indices = []
    
    print("Calculating similarities...")
    for layer in layers:
        if layer in refusal_directions:
            harm_dir = harmfulness_directions[layer]
            ref_dir = refusal_directions[layer]
            
            sim = compute_cosine_similarity(harm_dir, ref_dir)
            similarities.append(float(sim))
            layer_indices.append(layer)
            
    # Save results
    results = {
        "layers": layer_indices,
        "cosine_similarities": similarities
    }
    
    with open(output_path("results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to results.json")
    
    # 7. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, similarities, marker='o', linestyle='-', color='b')
    plt.title("Cosine Similarity between Harmfulness and Refusal Directions per Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.savefig(output_path("layer_wise_similarity.png"))
    print("Plot saved to layer_wise_similarity.png")
    
    # --- Deep Dive Extensions ---
    from probing import compute_norms, compute_projections
    from steering import SteeredModel
    
    # 8. Norm Analysis
    print("Computing norms...")
    harm_norms = compute_norms(harmfulness_directions)
    ref_norms = []
    # Refusal dir only exists for layers we computed it for (all?)
    for l in layers:
        if l in refusal_directions:
            ref_norms.append(torch.norm(refusal_directions[l]).item())
        else:
            ref_norms.append(0.0)
            
    plt.figure(figsize=(10, 6))
    plt.plot(layers, harm_norms, label="Harmfulness Norm", marker='o')
    plt.plot(layers, ref_norms, label="Refusal Norm", marker='x')
    plt.title("L2 Norm of Directions per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Norm")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path("layer_wise_norms.png"))
    print("Norms plot saved.")
    
    # 9. Projections Analysis (Layer 15 as example middle-late layer)
    target_layer = 15
    print(f"Computing projections at layer {target_layer}...")
    if target_layer in harmfulness_directions:
        direction = harmfulness_directions[target_layer]
        h_acts = harmful_acts_dict[target_layer]
        b_acts = harmless_acts_dict[target_layer]
        
        h_proj = compute_projections(direction, h_acts)
        b_proj = compute_projections(direction, b_acts)
        
        plt.figure(figsize=(8, 6))
        plt.hist(h_proj, alpha=0.5, label="Harmful", bins=10)
        plt.hist(b_proj, alpha=0.5, label="Harmless", bins=10)
        plt.title(f"Projections onto Harmfulness Direction (Layer {target_layer})")
        plt.legend()
        plt.savefig(output_path("projections_layer_15.png"))
        print("Projections plot saved.")
        
    # 10. Steering Experiment
    print("Running steering experiment...")
    steered_model = SteeredModel(model, tokenizer)
    
    # Try inserting refusal vector at layer 12
    steer_layer = 12
    if steer_layer in refusal_directions:
        refusal_vec = refusal_directions[steer_layer]
        
        test_prompts = [
            "How to bake a chocolate cake", 
            "Explain how photosynthesis works",
            "How to build a bomb" # Control: should be refused anyway
        ]
        formatted_test = [f"<|user|>\n{p} </s>\n<|assistant|>\n" for p in test_prompts]
        
        print("\n--- Baseline Generation ---")
        for p in formatted_test:
            print(f"Prompt: {p[-50:].strip()}") # Show end of prompt
            print(f"Response: {steered_model.generate(p)}")
            
        print(f"\n--- Steered Generation (Layer {steer_layer}, Coeff=1.5) ---")
        steered_model.add_steering_hook(steer_layer, refusal_vec, coeff=1.5)
        for p in formatted_test:
            print(f"Prompt: {p[-50:].strip()}")
            print(f"Response: {steered_model.generate(p)}")
            
        
        
        steered_model.clear_hooks()
        
    # --- Mechanistic Analysis: Attention ---
    from attention_analysis import get_head_contributions, plot_head_heatmap, visualize_attention_patterms
    
    print("\n--- Mechanistic Analysis: Attention ---")
    
    # 11. Attribution (Harmfulness Direction)
    # We use Harmonious/Harmful prompts for this? 
    # Let's see which heads write to the harmfulness direction when processing harmful prompts.
    print("Computing head contributions to Harmfulness Direction...")
    
    # Needs directions_dict (we have harmfulness_directions)
    # Needs prompts (formatted_harmful)
    
    # We'll use a subset to be fast?
    subset_harmful = formatted_harmful[:10]
    
    harm_contribs = get_head_contributions(model, tokenizer, harmfulness_directions, subset_harmful)
    top_harm_heads = plot_head_heatmap(harm_contribs, "Head Contributions to Harmfulness Direction", output_path("head_contrib_harmfulness.png"))
    
    print(f"Top Harmfulness Heads: {top_harm_heads}")
    
    # 12. Attribution (Refusal Direction)
    # Which heads write to the Refusal Direction?
    print("Computing head contributions to Refusal Direction...")
    # Typically calculated on harmful prompts where refusal is active? 
    # Or on the Contrastive difference? 
    # Let's check on the *Harmful Prompts* where the model *should* be refusing.
    
    ref_contribs = get_head_contributions(model, tokenizer, refusal_directions, subset_harmful)
    top_ref_heads = plot_head_heatmap(ref_contribs, "Head Contributions to Refusal Direction", output_path("head_contrib_refusal.png"))
    
    print(f"Top Refusal Heads: {top_ref_heads}")
    
    # 13. Visualizing Attention Patterns
    # Pick the top 1 head from interaction
    if top_ref_heads:
        top_ref_head = top_ref_heads[0] #(L, H)
        print(f"Visualizing attention pattern for Refusal Head {top_ref_head}...")
        
        sample_prompt = formatted_harmful[0] # "How to build a bomb..."
        sample_prompt = formatted_harmful[0] # "How to build a bomb..."
        visualize_attention_patterms(model, tokenizer, [top_ref_head], sample_prompt, filename_prefix=output_path("refusal_attn"))
        print("Attention pattern saved.")
        
    # --- Advanced Analysis & Validation ---
    print("\n--- Advanced Analysis & Validation ---")
    
    # 14. Verify Prompt Template
    print(f"Tokenizer chat template: {tokenizer.chat_template if tokenizer.chat_template else 'None'}")
    # Print a sample applied template
    sample_msg = [{"role": "user", "content": "Test"}]
    print(f"Sample Chat Template Application: {tokenizer.apply_chat_template(sample_msg, tokenize=False)}")
    
    # 15. Clustering (PCA)
    from clustering import generate_4way_data, extract_specific_activations, plot_pca_clustering
    
    # Use subset for clarity/speed? Or full? Full 20+20 is fine.
    print("Generating 4-way data for PCA clustering...")
    data_list = generate_4way_data(harmful_prompts, harmless_prompts)
    
    # Extract at Layer 15 (Harmfulness check) and Layer 21 (Refusal check)
    target_layers = [15, 20, 21] 
    # Note: Layer indices 0-21. So 21 is last layer block.
    
    print(f"Extracting activations for clustering at layers {target_layers}...")
    # This might take a moment on CPU
    cluster_results, labels, types, responses = extract_specific_activations(model, tokenizer, data_list, target_layers)
    
    for l in target_layers:
        plot_pca_clustering(cluster_results, labels, types, responses, l, output_path("pca_clustering"))
        
    # 16. Aggregated Attention
    from attention_analysis import visualize_aggregated_attention
    
    if top_ref_heads:
        top_head = top_ref_heads[0]
        print(f"Visualizing aggregated attention for {top_head} on 5 samples...")
        # Pick 5 harmful prompts
        samples = formatted_harmful[:5]
        visualize_aggregated_attention(model, tokenizer, top_head, samples, output_path("aggregated_attn_refusal.png"))

    # --- Layer-wise Ablation Study ---
    print("\n--- Layer-wise Ablation Study ---")
    from ablation import run_ablation_sweep, plot_ablation_effects
    
    # Ablation for Harmfulness Direction
    print("\nRunning ablation sweep for HARMFULNESS direction...")
    harm_effects, harm_baseline = run_ablation_sweep(
        model, tokenizer, formatted_harmful[:10], 
        harmfulness_directions, "Harmfulness"
    )
    harm_sorted = plot_ablation_effects(harm_effects, "Harmfulness", output_path("ablation_harmfulness.png"))
    
    # Ablation for Refusal Direction
    print("\nRunning ablation sweep for REFUSAL direction...")
    ref_effects, ref_baseline = run_ablation_sweep(
        model, tokenizer, formatted_harmful[:10],
        refusal_directions, "Refusal"
    )
    ref_sorted = plot_ablation_effects(ref_effects, "Refusal", output_path("ablation_refusal.png"))
    
    print("\n=== Ablation Study Complete ===")
    print("Saved: ablation_harmfulness.png, ablation_refusal.png")
    
    # --- Head-level Ablation Study ---
    print("\n--- Head-level Ablation Study ---")
    from head_ablation import run_head_ablation_study, plot_head_ablation_results
    
    # Top heads from attribution analysis
    # Harmfulness: L21H2, L21H7, L21H3, L20H6, L21H5
    # Refusal: L21H3, L20H28, L20H6, L21H21, L21H24
    
    # Combine unique heads to test
    heads_to_test = [
        (21, 2), (21, 7), (21, 3), (20, 6), (21, 5),  # Top harmfulness
        (20, 28), (21, 21), (21, 24),  # Top refusal (some overlap)
        (21, 0), (21, 1), (20, 0), (19, 0),  # Control heads for comparison
    ]
    
    # Head ablation for Harmfulness
    print("\nHead ablation for HARMFULNESS direction...")
    harm_head_results, harm_base = run_head_ablation_study(
        model, tokenizer, formatted_harmful[:10],
        harmfulness_directions, heads_to_test, "Harmfulness"
    )
    plot_head_ablation_results(harm_head_results, "Harmfulness", output_path("head_ablation_harmfulness.png"))
    
    # Head ablation for Refusal
    print("\nHead ablation for REFUSAL direction...")
    ref_head_results, ref_base = run_head_ablation_study(
        model, tokenizer, formatted_harmful[:10],
        refusal_directions, heads_to_test, "Refusal"
    )
    plot_head_ablation_results(ref_head_results, "Refusal", output_path("head_ablation_refusal.png"))
    
    print("\n=== Head Ablation Complete ===")
    print("Saved: head_ablation_harmfulness.png, head_ablation_refusal.png")

if __name__ == "__main__":
    main()
