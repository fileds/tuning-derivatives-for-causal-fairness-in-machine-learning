import torch

def compute_gradient(model, X):
    # Convert the data to PyTorch tensors
    X = X.clone().detach().requires_grad_(True)
    pred = model(X).squeeze()
    gradient = torch.autograd.grad(pred, X, grad_outputs=torch.ones_like(pred), create_graph=True)[0].detach()

    return gradient