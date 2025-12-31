import torch

class SteeredModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        
    def add_steering_hook(self, layer_idx, steering_vector, coeff=1.0):
        """
        Adds a forward hook to add the steering vector to the hidden states.
        """
        steering_vector = steering_vector.to(self.model.device)
        
        def hook(module, input, output):
            # output is tuple (hidden_states,)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Add vector to the last token position? Or all positions?
            # Usually steering is applied to all positions for simplicity in generation,
            # or specifically at the instruction end. 
            # For "Refusal", applying it to all tokens (system/user prompt) primes the state.
            
            # hidden_states: (batch, seq_len, dim)
            # vector: (dim,)
            
            # Broadcasting adds to all tokens
            hidden_states += coeff * steering_vector
            
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states
                
        # Register hook on the specific layer
        # Transformers layer naming: model.layers[i] (usually)
        # Checking architecture for TinyLlama (LlamaForCausalLM) -> model.model.layers[i]
        
        if hasattr(self.model, "model"):
            layer_module = self.model.model.layers[layer_idx]
        elif hasattr(self.model, "layers"):
            layer_module = self.model.layers[layer_idx]
        else:
            raise ValueError("Could not find layers module in model structure")
            
        handle = layer_module.register_forward_hook(hook)
        self.hooks.append(handle)
        
    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        
    def generate(self, prompt, max_new_tokens=20):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False # Greedy for deterministic check
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
