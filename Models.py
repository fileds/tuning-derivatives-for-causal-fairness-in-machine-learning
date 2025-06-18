import torch
import torch.nn as nn

class NeuralRegressor(nn.Module):
    def __init__(self, n_features, hidden_layers, activation=nn.ELU()):
        super(NeuralRegressor, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(n_features, hidden_layers[0]))
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, X):
        for layer in self.layers[:-1]:
            X = self.activation(layer(X))
        return self.layers[-1](X)

    def predict(self, X):
        with torch.no_grad():
            return self.forward(X).detach().numpy()

class MeanMarginalizedRegressor:
    def __init__(self, model, protected_idx, protected_mean):
        self.model = model
        self.protected_idx = protected_idx
        self.protected_mean = protected_mean

    def forward(self, X):
        # Detach to stop tracking gradient 
        X_mod = X.clone()

        # Replace protected features with mean
        for idx, mean_value in zip(self.protected_idx, self.protected_mean):
            X_mod[:, idx] = mean_value

        # Predict
        output = self.model(X_mod)

        return output

    def predict(self, X):
        return self.forward(X).detach().numpy()

    def __call__(self, X):
        return self.forward(X)

class NeuralClassifier(nn.Module):
    def __init__(self, n_features, hidden_layers, activation=nn.ELU()):
        super(NeuralClassifier, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(n_features, hidden_layers[0]))
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], 1))

    def forward(self, X):
        for layer in self.layers[:-1]:
            X = self.activation(layer(X))
        return self.layers[-1](X)

    def predict_proba(self, X):
        with torch.no_grad():
            return nn.Sigmoid()(self(X)).detach().numpy().squeeze()

    def predict(self, X, threshold=0.5):
        with torch.no_grad():
            return self.predict_proba(X) >= threshold

class MarginalizedClassifier:
    def __init__(self, model, protected_idx, protected_values):
        self.model = model
        self.protected_idx = protected_idx
        self.protected_values = protected_values

    def forward(self, X):
        # Clone to stop tracking gradient 
        X_mod = X.clone()
        for idx, value in zip(self.protected_idx, self.protected_values):
            X_mod[:, idx] = value

        return self.model(X_mod)

    def predict_proba(self, X):
        with torch.no_grad():
            return nn.Sigmoid()(self.forward(X)).detach().numpy().squeeze()

    def predict(self, X, threshold=0.5):
        with torch.no_grad():
            return self.predict_proba(X) >= threshold

    def __call__(self, X):
        return self.forward(X)