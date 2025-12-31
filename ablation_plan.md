# Ablation Study: Causal Responsibility of Layers

## Goal
For each layer L (0-21), ablate (remove) the Harmfulness/Refusal direction from the residual stream at that layer and measure the downstream effect.

## Methodology

### Direction Ablation
At layer L, we project out the direction from the hidden state:
```
h_ablated = h - (h · d_unit) * d_unit
```
where `d_unit` is the unit vector of the direction.

### Metrics

**For Harmfulness Ablation:**
- Measure projection onto Harmfulness direction at Layer 21 (final)
- If ablating layer L removes harmfulness info, the final projection should drop

**For Refusal Ablation:**
- Generate response and check for refusal keywords
- Or measure projection onto Refusal direction at Layer 21

### Output
- Plot: "Effect of Ablating Harmfulness Dir at Layer X" on final Harmfulness projection
- Plot: "Effect of Ablating Refusal Dir at Layer X" on refusal behavior
- Identify critical layers where ablation has maximum effect

## Implementation Plan

1. Create `ablation.py` with:
   - `ablate_direction_hook(layer_idx, direction)` - hook that projects out direction
   - `measure_final_projection(model, tokenizer, prompts, direction, ablation_layer)`
   - `run_ablation_sweep(model, tokenizer, prompts, direction_dict)`

2. Update `main.py` to run ablation experiments

3. Generate plots showing causal importance per layer
