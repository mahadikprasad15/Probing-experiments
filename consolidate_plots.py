import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

def consolidate():
    images = [
        "layer_wise_similarity.png",
        "layer_wise_norms.png",
        "projections_layer_15.png",
        "head_contrib_refusal.png",
        "refusal_attn_L21_H3.png",
        "pca_clustering_L15.png",
        "pca_clustering_L21.png",
        "aggregated_attn_refusal.png"
    ]
    
    titles = [
        "1. Orthogonality",
        "2. Trends: Norms",
        "3. Separation: Projections",
        "4. Mech: Refusal Heatmap",
        "5. Mech: Pattern (One)",
        "6. Validation: PCA L15",
        "7. Validation: PCA L21 (Refusal)",
        "8. Validation: Aggregated Attn"
    ]
    
    # Create a figure with 2 rows and 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for i, img_name in enumerate(images):
        img_path = os.path.join(OUTPUT_DIR, img_name)
        try:
            img = mpimg.imread(img_path)
            axes[i].imshow(img)
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].axis('off')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            axes[i].text(0.5, 0.5, "Image not found", ha='center')
            axes[i].axis('off')
            
    # Remove last empty subplot if any
    if len(images) < len(axes):
        for j in range(len(images), len(axes)):
            axes[j].axis('off')
            
    plt.tight_layout()
    output_file = os.path.join(OUTPUT_DIR, "consolidated_analysis.png")
    plt.savefig(output_file, dpi=150)
    print(f"Saved {output_file}")

if __name__ == "__main__":
    consolidate()
