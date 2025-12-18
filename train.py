"""
STUDENT IMPLEMENTATION REQUIRED

This file contains the training loop that you need to implement for HW1.
You should complete the train_model function by implementing the training logic
including optimizer setup, loss function, training loop, and model saving.

TODO: Implement the training loop in the train_model function
"""

# define your training loop here
import torch
from eval import evaluate_metrics

def train_model(model, predict_fn, train_loader, val_loader, device, save_path='best_model.pt', lr=0.001):
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
     

    best_val_f1 = float('-inf')
    # Early stopping parameters
    patience = 10 
    patience_counter = 0
    best_model_state = None

    train_losses = [] 
    val_losses = []

    for epoch in range(500):
        # TRAINING PHASE
        model.train()
        running_train_loss = 0.0

        for train_x, train_y in train_loader:
            train_x, train_y = train_x.to(device), train_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(train_x)
            loss = criterion(outputs, train_y.float())
            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()


        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        running_val_loss = 0.0
    
        with torch.no_grad():
            # VALIDATION PHASE
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                
                outputs = model(val_x).squeeze(1)
                loss = criterion(outputs, val_y.float())
                running_val_loss += loss.item()

            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Evaluate metrics on the validation set
            f1_weighted, precision, recall, f1_micro, support = evaluate_metrics(
                model, val_loader, predict_fn, device
            )
        
        if f1_weighted > best_val_f1:
            best_val_f1 = f1_weighted
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save(best_model_state, save_path)
            print(f'*** Improved F1. Model saved at {save_path} ***')
        else:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # save best model (so far)
    torch.save(model.state_dict(), save_path)

    # load best model from training run
    model.load_state_dict(torch.load(save_path))
    f1_weighted, precision, recall, f1, support = evaluate_metrics(model, val_loader, predict_fn, device)
    print(f"*** Best (weighted) F1: {f1_weighted} ***")
    print(f'*** Best model weights saved at {save_path} ***')
    
    return model, train_losses, val_losses
    