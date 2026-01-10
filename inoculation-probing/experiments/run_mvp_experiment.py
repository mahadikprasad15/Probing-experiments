"""
MVP Experiment: Test if inoculation prompts improve probe generalization.

This script runs the minimal viable experiment to test the hypothesis:
"Inoculation prompts can disentangle spurious correlations and improve probe generalization"
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SpuriousDatasetBuilder
from src.extraction import ActivationExtractor
from src.probing import ProbeExperiment
from src.prompts import format_prompt, get_mvp_prompts
from src.utils import ExperimentLogger


def run_mvp_experiment(
    # Dataset config
    n_train_per_group: int = 100,
    n_test_per_group: int = 50,
    short_range: tuple = (10, 30),
    long_range: tuple = (100, 200),

    # Model config
    model_name: str = 'meta-llama/Llama-3.2-1B',
    layers: list = [12, 20],

    # Experiment config
    prompts: list = None,
    seed: int = 42,

    # Paths (for Colab with Google Drive)
    base_dir: Path = None,
    use_cache: bool = True
):
    """
    Run MVP experiment.

    Args:
        n_train_per_group: Number of training samples per group
        n_test_per_group: Number of test samples per group
        short_range: Token range for short reviews
        long_range: Token range for long reviews
        model_name: HuggingFace model name
        layers: List of layers to probe
        prompts: List of prompt names to test (None = MVP default)
        seed: Random seed
        base_dir: Base directory for saving (None = current dir)
        use_cache: Whether to cache activations
    """
    # Setup paths
    if base_dir is None:
        base_dir = Path('./results')
    else:
        base_dir = Path(base_dir)

    dataset_dir = base_dir / 'datasets' / 'mvp_dataset'
    activations_dir = base_dir / 'activations'
    results_dir = base_dir / 'experiments' / 'mvp_experiment'

    # Setup logger
    logger = ExperimentLogger(results_dir)

    # Get prompts to test
    if prompts is None:
        prompts = get_mvp_prompts()

    print("="*80)
    print("MVP INOCULATION PROBING EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Dataset: {n_train_per_group*2} train, {n_test_per_group*2} test")
    print(f"  Model: {model_name}")
    print(f"  Layers: {layers}")
    print(f"  Prompts: {prompts}")
    print(f"  Results: {results_dir}")
    print()

    # -------------------------------------------------------------------------
    # STEP 1: Create Dataset
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: CREATE SPURIOUSLY CORRELATED DATASET")
    print("="*80)

    if (dataset_dir / 'metadata.json').exists():
        print(f"Loading existing dataset from {dataset_dir}")
        dataset = SpuriousDatasetBuilder.load(dataset_dir)
    else:
        print(f"Creating new dataset...")
        dataset = SpuriousDatasetBuilder(
            source_dataset='imdb',
            train_correlation=1.0,  # Perfect spurious correlation
            n_train_per_group=n_train_per_group,
            n_test_per_group=n_test_per_group,
            short_range=short_range,
            long_range=long_range,
            seed=seed
        )

        splits = dataset.create_splits()
        dataset.save(dataset_dir)

    train_data = dataset.train_data
    test_data = dataset.test_data

    # -------------------------------------------------------------------------
    # STEP 2: Sanity Checks
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: SANITY CHECKS")
    print("="*80)

    # TODO: Add sanity checks
    # - Length probe (should be ~100% accurate)
    # - Sentiment probe on balanced data (should be high)

    print("Sanity checks: SKIPPED (add if needed)")

    # -------------------------------------------------------------------------
    # STEP 3: Run Experiments for Each Prompt & Layer
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: RUN EXPERIMENTS")
    print("="*80)

    all_results = []

    for layer in layers:
        print(f"\n{'='*80}")
        print(f"LAYER {layer}")
        print(f"{'='*80}")

        # Initialize extractor for this layer
        extractor = ActivationExtractor(
            model_name=model_name,
            layer=layer,
            cache_dir=activations_dir if use_cache else None
        )

        for prompt_name in prompts:
            print(f"\n{'-'*80}")
            print(f"Prompt: {prompt_name}")
            print(f"{'-'*80}")

            # Start logging run
            run_name = f"layer{layer}_{prompt_name}"
            config = {
                'layer': layer,
                'prompt_name': prompt_name,
                'model_name': model_name,
                'n_train': len(train_data['texts']),
                'n_test': len(test_data['texts']),
                'seed': seed
            }
            logger.start_run(run_name, config)

            # Format texts with prompt
            train_texts_prompted = [
                format_prompt(prompt_name, text)
                for text in train_data['texts']
            ]
            test_texts_prompted = [
                format_prompt(prompt_name, text)
                for text in test_data['texts']
            ]

            # Extract activations
            print("Extracting train activations...")
            activations_train = extractor.extract(
                train_texts_prompted,
                prompt_name=f"{prompt_name}_train",
                use_cache=use_cache
            )

            print("Extracting test activations...")
            activations_test = extractor.extract(
                test_texts_prompted,
                prompt_name=f"{prompt_name}_test",
                use_cache=use_cache
            )

            # Create probe experiment
            metadata_test = {
                'length_labels': test_data['length_labels'],
                'sentiment_labels': test_data['sentiment_labels']
            }

            probe_exp = ProbeExperiment(
                activations_train=activations_train,
                labels_train=np.array(train_data['sentiment_labels']),
                activations_test=activations_test,
                labels_test=np.array(test_data['sentiment_labels']),
                metadata_test=metadata_test
            )

            # Train probe
            probe_exp.train_probe(regularization='cv')

            # Evaluate
            metrics = probe_exp.evaluate(n_bootstrap=100)

            # Visualize
            viz_path = results_dir / f"viz_{run_name}.png"
            probe_exp.visualize_results(save_path=viz_path)

            # Save probe
            probe_dir = results_dir / 'probes' / run_name
            probe_exp.save(probe_dir)

            # Log results
            logger.log_metrics(metrics)
            logger.log_artifact('visualization', viz_path)
            logger.log_artifact('probe', probe_dir / 'probe_model.pkl')
            logger.end_run()

            # Store for summary
            all_results.append({
                'layer': layer,
                'prompt': prompt_name,
                **metrics
            })

    # -------------------------------------------------------------------------
    # STEP 4: Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    # Create summary table
    print("\nResults by Layer and Prompt:")
    print(f"{'Layer':<8} {'Prompt':<25} {'Overall Acc':<12} {'Worst-Group Acc':<15}")
    print("-" * 70)

    for result in all_results:
        layer = result['layer']
        prompt = result['prompt']
        overall_acc = result['accuracy_overall']
        worst_acc = result.get('accuracy_worst_group', 0.0)

        print(f"{layer:<8} {prompt:<25} {overall_acc:.4f}       {worst_acc:.4f}")

    # Find best prompt per layer
    print("\n\nBest Prompts by Layer (by worst-group accuracy):")
    for layer in layers:
        layer_results = [r for r in all_results if r['layer'] == layer]
        best = max(layer_results, key=lambda x: x.get('accuracy_worst_group', 0))

        print(f"\nLayer {layer}:")
        print(f"  Best prompt: {best['prompt']}")
        print(f"  Overall accuracy: {best['accuracy_overall']:.4f}")
        print(f"  Worst-group accuracy: {best.get('accuracy_worst_group', 0):.4f}")

    # Compare inoculation vs baseline
    print("\n\nInoculation Effect Analysis:")
    for layer in layers:
        layer_results = [r for r in all_results if r['layer'] == layer]

        baseline = next((r for r in layer_results if r['prompt'] == 'baseline'), None)
        inoc_results = [r for r in layer_results if 'inoc' in r['prompt']]

        if baseline and inoc_results:
            baseline_acc = baseline.get('accuracy_worst_group', baseline['accuracy_overall'])

            print(f"\nLayer {layer}:")
            print(f"  Baseline: {baseline_acc:.4f}")

            for inoc in inoc_results:
                inoc_acc = inoc.get('accuracy_worst_group', inoc['accuracy_overall'])
                improvement = inoc_acc - baseline_acc

                print(f"  {inoc['prompt']}: {inoc_acc:.4f} ({improvement:+.4f})")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")

    return all_results


if __name__ == '__main__':
    # Run MVP experiment with default settings
    results = run_mvp_experiment()
