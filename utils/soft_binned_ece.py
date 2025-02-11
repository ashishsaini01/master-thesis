import torch
import torch.nn.functional as F

EPS = 1e-5

def compute_squared_error_label_binning_pytorch(logits, y, m, temperature):
    """
    Computes and returns the squared soft-binned ECE (label-binned) tensor.
    
    Soft-binned ECE (label-binned, l2-norm) is defined in equation (12) in this
    paper: https://arxiv.org/abs/2108.00106. It is a softened version of ECE
    (label-binned) which is defined in equation (7).
    
    Args:
        logits: tensor of predicted logits of (batch-size, num-classes) shape
        y: tensor of labels in [0,num-classes) of (batch-size,) shape
        m: number of bins
        temperature: soft binning temperature

    Returns:
        A tensor containing a single value: squared soft-binned ECE.
    """
    
    # Convert logits and y to torch tensors
    logits = torch.tensor(logits)
    y = torch.tensor(y)
    
    # Calculate softmax probabilities
    q_hat = F.softmax(logits, dim=1)
    
    # Get predicted labels
    _, y_hat = torch.max(q_hat, dim=1)
    
    # Get maximum probabilities
    p_hat = torch.max(q_hat, dim=1)[0]
    
    # Calculate a, indicator function for correct predictions
    a = torch.eq(y_hat, y).float()
    
    # Define bins
    bins = torch.linspace(1.0 / logits.shape[1], 1.0, m + 1)
    b = (bins[1:] + bins[:-1]) * 0.5
    
    # Calculate soft binning numerator and denominator
    c_numerator = torch.exp(-(b.view(1, -1) - p_hat.view(-1, 1)) ** 2 / temperature)
    c_denominator = c_numerator.sum(dim=1)
    
    # Soft binning
    c = (1 - EPS) * (c_numerator / c_denominator.view(-1, 1)) + EPS / m
    
    # Calculate a_bar
    a_bar_numerator = torch.matmul(c.t(), a)
    a_bar_denominator = c.sum(dim=0)
    a_bar = a_bar_numerator / a_bar_denominator
    
    # Calculate squared error
    squared_error = torch.sum(c * (a_bar.view(1, -1) - p_hat.view(-1, 1)) ** 2)
    
    # Scale by batch size
    squared_error_scaled = squared_error / logits.shape[0]
    
    return squared_error_scaled.item()