import sys
import torch
import torch.nn as nn
import torch.optim as optim

def fair_tuning(
    target_model, 
    train_loader, 
    optimizer,
    prediction_criterion,
    sp_idx, 
    pp_idx, 
    n_epochs, 
    alpha_spd, 
    alpha_ppd, 
    store_losses=True,
    verbose=False
):
    # Loss criteria
    sp_criterion = nn.L1Loss()
    pp_criterion = nn.L1Loss()

    if store_losses:
        losses = {
            'total': [],
            'sp': [],
            'pp': [],
            'prediction': []
        }
    target_model.train()
    for epoch in range(n_epochs):
        if store_losses:
            train_losses = {
                'total': 0.0,
                'sp': 0.0,
                'pp': 0.0,
                'prediction': 0.0
            }
        for X_batch, gradient_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # Predictions and gradients of target model
            y_pred = target_model(X_batch)
            gradient = torch.autograd.grad(
                y_pred, X_batch, 
                grad_outputs=torch.ones_like(y_pred), 
                create_graph=True
            )[0]
            # Prediction loss
            prediction_loss = prediction_criterion(y_pred, y_batch)
            # Statistical parity loss
            sp_loss = sp_criterion(
                gradient[:, sp_idx], 
                torch.zeros_like(gradient[:, sp_idx])
            )
            # Predictive parity loss
            pp_loss = pp_criterion(
                gradient[:, pp_idx], 
                gradient_batch[:, pp_idx]
            )

            loss = prediction_loss + alpha_spd * sp_loss + alpha_ppd * pp_loss
            loss.backward()
            optimizer.step()

            if store_losses:
                train_losses['total'] += loss.item()
                train_losses['sp'] += sp_loss.item()
                train_losses['pp'] += pp_loss.item()
                train_losses['prediction'] += prediction_loss.item()

        if store_losses and verbose and (epoch+1) % 10 == 0:
            # Print epoch training loss
            sys.stdout.write(
                f"\rEpoch {epoch+1}/{n_epochs} | "
                f"Total Loss: {train_losses['total']/len(train_loader):.4f} | "
                f"SP Loss: {train_losses['sp']/len(train_loader):.4f} | "
                f"PP Loss: {train_losses['pp']/len(train_loader):.4f} | "
                f"Prediction Loss: {train_losses['prediction']/len(train_loader):.4f}"
            )
            sys.stdout.flush()

        if store_losses:
            losses['total'].append(train_losses['total']/len(train_loader))
            losses['sp'].append(train_losses['sp']/len(train_loader))
            losses['pp'].append(train_losses['pp']/len(train_loader))
            losses['prediction'].append(train_losses['prediction']/len(train_loader))

    if store_losses and verbose:
        return target_model, losses
    else: 
        return target_model
