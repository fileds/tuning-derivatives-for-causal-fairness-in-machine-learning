import numpy as np
import pandas as pd
import torch 

from DGPs.DGP import DGP

class LinearHiringDGP(DGP):
    def __init__(self, effects, constants):
        super().__init__(effects)

        self.constants = constants

    def generate_data(self, num_samples, Y_noise_sd=1, return_tensors=False):
        # Background
        U = np.random.normal(0, 1, num_samples)
        # Generate linear X, Z, W
        X = self.effects[('U', 'X')] * U + np.random.normal(0, 1, num_samples) + self.constants['X']
        Z = self.effects[('U', 'Z')] * U + np.random.normal(0, 1, num_samples) + self.constants['Z']
        W = (
            self.effects[('X', 'W')] * X 
            + self.effects[('Z', 'W')] * Z 
            + np.random.normal(0, 1, num_samples)
            + self.constants['W']
        )

        # Generate non-linear Y
        U_Y = np.random.normal(0, Y_noise_sd, num_samples)
        # Create a probability of Y being 1
        prob_Y = 1 / (1 + np.exp(-(
            self.effects[('X', 'Y')] * X
            + self.effects[('Z', 'Y')] * Z
            + self.effects[('W', 'Y')] * W
            + U_Y
        )))
        # Generate Y based on the probability
        Y = np.random.binomial(1, prob_Y, num_samples)

        df = pd.DataFrame({
            'X': X,
            'Z': Z,
            'W': W,
            'Y': Y
        })

        X = df[['X', 'Z', 'W']]
        y = df['Y']

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

        if return_tensors:
            return df, X_tensor, y_tensor
        else: 
            return df

    def compute_gradient(self, X_train_tensor):
        num_samples = X_train_tensor.shape[0]
        # Compute gradients
        gradient = np.array([
            np.repeat(self.effects[('X', 'Y')], num_samples),
            np.repeat(self.effects[('Z', 'Y')], num_samples),
            np.repeat(self.effects[('W', 'Y')], num_samples)
        ]).T

        return gradient