import torch
from torch import nn
from collections import OrderedDict # Useful for state_dict manipulation later

# --- 4. Neural Network Models ---

class FcReluDnn(nn.Module):
    """
    Fully-Connected ReLU Deep Neural Network.
    This class defines the neural network architecture.
    """
    def __init__(self, vLayers: list):
        """
        Constructor for the FcReluDnn model.
        
        Args:
            vLayers (list): A list of integers defining the number of neurons in each layer.
                            Example: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim].
        """
        super(FcReluDnn, self).__init__()
        
        self.hidden = nn.ModuleList() # Use ModuleList to store linear layers
        
        # Create linear layers with ReLU activation for hidden layers
        # and a final linear layer for the output (no activation here, softmax applied later).
        for l_idx, (input_size, output_size) in enumerate(zip(vLayers, vLayers[1:])):
            # All layers use torch.float64 as specified in the original code for Hessian calculations
            linear_layer = nn.Linear(input_size, output_size, dtype=torch.float64)
            self.hidden.append(linear_layer)
        
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            activation (torch.Tensor): Input tensor to the network.
        
        Returns:
            torch.Tensor: Output tensor (logits before softmax).
        """
        L = len(self.hidden) # Number of layers
        
        for l_idx, linear_transform in enumerate(self.hidden):
            activation = linear_transform(activation)
            # Apply ReLU activation for all hidden layers (not the last output layer)
            if l_idx < L - 1:
                activation = torch.nn.functional.relu(activation)
        return activation

    def num_parameters(self) -> int:
        """
        Calculates the total number of trainable parameters in the model.
        
        Returns:
            int: Total number of parameters.
        """
        return sum(torch.numel(w) for w in self.parameters())


class FcReluDnn_external(nn.Module):
    """
    Fully-Connected ReLU Deep Neural Network designed to operate with externally provided parameters.
    This is used when model parameters are managed and passed explicitly (e.g., for Hessian calculations).
    """
    def __init__(self):
        """
        Constructor for the FcReluDnn_external model.
        It does not initialize its own nn.Linear layers, as parameters are external.
        """
        super(FcReluDnn_external, self).__init__()
        # Note: No need to initialize nn.Linear layers here since parameters are provided externally.
        # This module will be used to apply linear transformations and activations given external weights/biases.
        
    def forward(self, net_in: torch.Tensor, net_params: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using externally provided network parameters.
        
        Args:
            net_in (torch.Tensor): Input tensor to the network.
            net_params (list[torch.Tensor]): A list of tensors representing the weights and biases of all layers.
                                            Assumed to be ordered as [weight1, bias1, weight2, bias2, ...].
        
        Returns:
            torch.Tensor: Output tensor (logits before softmax).
        """
        # Calculate number of layers from the parameter list (each layer has weight and bias)
        L = len(net_params) // 2 
        
        for ll in range(L):
            curr_layer_weight = net_params[2 * ll]
            curr_layer_bias = net_params[2 * ll + 1]
            
            # Apply linear transformation using functional API
            net_in = torch.nn.functional.linear(net_in, curr_layer_weight, curr_layer_bias)
            
            # Apply ReLU activation for all hidden layers (not the last output layer)
            if ll < L - 1: 
                net_in = torch.nn.functional.relu(net_in)
        return net_in

# --- Example Usage (for testing purposes) ---
if __name__ == '__main__':
    print("--- Testing FcReluDnn ---")
    input_dim = 2
    output_dim = 8
    hidden_dims = [10, 30, 30]
    
    # Define layer sizes for FcReluDnn
    v_layers_test = [input_dim] + hidden_dims + [output_dim]
    
    # Create an instance of the model
    model_test = FcReluDnn(v_layers_test)
    print(f"Model architecture: {model_test}")
    print(f"Total number of parameters: {model_test.num_parameters()}")
    
    # Create a dummy input tensor
    dummy_input = torch.randn(5, input_dim, dtype=torch.float64) # 5 samples, input_dim features
    
    # Perform a forward pass
    output_logits = model_test(dummy_input)
    print(f"Output logits shape: {output_logits.shape}") # Should be [5, output_dim]
    print(f"Output logits (first sample): {output_logits[0]}")

    print("\n--- Testing FcReluDnn_external ---")
    # To test FcReluDnn_external, we need to extract parameters from a regular model
    external_model_test = FcReluDnn_external()
    
    # Get parameters from a trained (or initialized) model
    # Convert parameters to a list of tensors for FcReluDnn_external
    model_params_list = [p.data for p in model_test.parameters()] 
    
    # Perform a forward pass using external parameters
    output_logits_external = external_model_test(dummy_input, model_params_list)
    print(f"Output logits (from external model) shape: {output_logits_external.shape}")
    print(f"Output logits (first sample from external): {output_logits_external[0]}")

    # Verify that both models produce the same output with the same parameters
    print(f"Outputs are identical: {torch.allclose(output_logits, output_logits_external)}")