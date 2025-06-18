import numpy as np
import pandas as pd
import torch 

from DGPs.DGP import DGP

class MultiplicativeDGP(DGP):
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
        Y = (
            self.effects[('X', 'Y')] * X
            * self.effects[('Z', 'Y')] * Z
            * self.effects[('W', 'Y')] * W
            + U_Y + self.constants['Y']
        )

        df = pd.DataFrame({
            'X': X,
            'Z': Z,
            'W': W,
            'Y': Y
        })

        if return_tensors:
            X = df[['X', 'Z', 'W']]
            y = df['Y']

            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            y_tensor = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

            return df, X_tensor, y_tensor
        else: 
            return df

    def compute_gradient(self, data, data_frame=False):
        if data_frame:
            data = data[['X', 'Z', 'W']].values
        X = data[:, 0]
        Z = data[:, 1]
        W = data[:, 2]
         
        dYdX = (
            self.effects[('X', 'Y')]
            * self.effects[('Z', 'Y')] * Z
            * self.effects[('W', 'Y')] * W
        )
        dYdZ = (
            self.effects[('X', 'Y')] * X
            * self.effects[('Z', 'Y')]
            * self.effects[('W', 'Y')] * W
        )
        dYdW = (
            self.effects[('X', 'Y')] * X
            * self.effects[('Z', 'Y')] * Z
            * self.effects[('W', 'Y')]
        )

        gradient = np.array([dYdX, dYdZ, dYdW]).T

        return gradient