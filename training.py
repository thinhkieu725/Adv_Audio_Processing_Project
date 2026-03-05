from copy import deepcopy
import numpy as np
from torch import cuda, no_grad
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

from model import ClassifierModel
from loss import HierarchicalLoss
from dataloader import Dataloaders
### IMPORT DATASET AND/OR DATALOADER HERE


def main():
    # Check if MPS or CUDA is available, else use CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Process on {device}', end='\n\n')

    # Define hyper-parameters to be used.
    # Training hyperparameters
    epochs = 50
    patience = 12
    learning_rate = 1e-3
    batch_size = 32
    best_checkpoint_path = 'best_model.pt'

    # Model hyperparameters
    num_parent_classes = 5
    num_child_classes = 23
    hidden_dim = 256
    dropout = 0.2
    BEATs_checkpoint_path = './BEATs_iter3_plus_AS2M.pt'

    # Loss hyperparameters
    lambda_parent = 1.0
    lambda_leaf = 1.0

    # Instantiate our DNN
    model = ClassifierModel(
        num_parent_classes=num_parent_classes,
        num_leaf_classes=num_child_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        BEATs_checkpoint_path=BEATs_checkpoint_path,
    )
    # Pass DNN to the available device.
    model = model.to(device)

    # Define the optimizer and give the parameters of the CNN model to an optimizer.
    optimizer = Adam(params=model.parameters(), lr=learning_rate)

    # Instantiate the loss function as a class.
    loss_function = HierarchicalLoss(
        lambda_parent=lambda_parent,
        lambda_leaf=lambda_leaf
    )

    train_loader, valid_loader, test_loader = Dataloaders(
        clean_csv_dir= 'bsd10k-splits', 
        noisy_csv_path='noisy_data/bsd35k-train-14k.csv', 
        clean_audio_dir='clean_data/audio', 
        noisy_audio_dir='noisy_data/audio', 
        batch_size=16
    )
    # Dummy data loaders for testing purposes. --- IGNORE ---
    num_batches = 10
    train_loader = [(torch.randn(batch_size, 16000), torch.zeros(batch_size, 16000).bool(), torch.randint(0, num_parent_classes, (batch_size,)), torch.randint(0, num_child_classes, (batch_size,))) for _ in range(num_batches)]
    valid_loader = [(torch.randn(batch_size, 16000), torch.zeros(batch_size, 16000).bool(), torch.randint(0, num_parent_classes, (batch_size,)), torch.randint(0, num_child_classes, (batch_size,))) for _ in range(num_batches)]
    test_loader = [(torch.randn(batch_size, 16000), torch.zeros(batch_size, 16000).bool(), torch.randint(0, num_parent_classes, (batch_size,)), torch.randint(0, num_child_classes, (batch_size,))) for _ in range(num_batches)]

    # Variables for the early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0

    best_model = None

    # Start training.
    for epoch in range(epochs):

        # Lists to hold the corresponding losses of each epoch.
        epoch_loss_training = []
        epoch_loss_validation = []

        # Indicate that we are in training mode, so (e.g.) dropout
        # will function
        model.train()

        # For each batch of our dataset.
        for batch in train_loader:
            
            # Zero the gradient of the optimizer.
            optimizer.zero_grad()

            # Get the batches.
            x, padding_mask, y_parent, y_leaf = batch
                   
            # Give them to the appropriate device.
            x = x.to(device)
            padding_mask = padding_mask.to(device)
            y_parent = y_parent.to(device)
            y_leaf = y_leaf.to(device)

            # Get the predictions .
            y_hat_parent, y_hat_leaf = model(x, padding_mask)

            # Calculate the loss .
            loss = loss_function(y_hat_parent, y_hat_leaf, y_parent, y_leaf)

            # Do the backward pass
            loss.backward()

            # Do an update of the weights (i.e. a step of the optimizer)
            optimizer.step()

            # Append the loss of the batch
            epoch_loss_training.append(loss.item())

        # Indicate that we are in evaluation mode
        model.eval()

        # Say to PyTorch not to calculate gradients, so everything will
        # be faster.
        with no_grad():

            # For every batch of our validation data.
            for batch in valid_loader:
                # Get the batch
                x_val, padding_mask_val, y_parent_val, y_leaf_val = batch
                
                # Pass the data to the appropriate device.
                x_val = x_val.to(device)
                padding_mask_val = padding_mask_val.to(device)
                y_parent_val = y_parent_val.to(device)
                y_leaf_val = y_leaf_val.to(device)
                # Get the predictions of the model.
                y_hat_parent, y_hat_leaf = model(x_val, padding_mask_val)

                # Calculate the loss.
                loss = loss_function(y_hat_parent, y_hat_leaf, y_parent_val, y_leaf_val)

                # Append the validation loss.
                epoch_loss_validation.append(loss.item())

        # Calculate mean losses.
        epoch_loss_validation = np.array(epoch_loss_validation).mean()
        epoch_loss_training = np.array(epoch_loss_training).mean()

        # Check early stopping conditions.
        if epoch_loss_validation < lowest_validation_loss:
            lowest_validation_loss = epoch_loss_validation
            patience_counter = 0
            best_model = deepcopy(model.state_dict())
            best_validation_epoch = epoch
        else:
            patience_counter += 1

        # If we have to stop, do the testing.
        if (patience_counter >= patience) or (epoch==epochs-1):
            
            print('\nExiting due to early stopping', end='\n\n')
            print(f'Best epoch {best_validation_epoch} with loss {lowest_validation_loss}', end='\n\n')
            if best_model is None:
                print('No best model. ')
            else:
                # Output checkpoint of the best model for later inspection.
                torch.save({
                    'epoch': best_validation_epoch,
                    'validation_loss': lowest_validation_loss,
                    'model_state_dict': best_model,
                }, best_checkpoint_path)

                # ROUGHLY TESTING PROCEDURE
                # Process similar to validation.
                print('Starting testing', end=' | ')
                testing_loss = []
                # Load best model 
                model.load_state_dict(best_model)
                model.eval()

                batch_accuracies = []
                
                with no_grad():
                    for batch in test_loader:
                        x_test, padding_mask_test, y_parent_test, y_leaf_test = batch

                        # Pass the data to the appropriate device.
                        x_test = x_test.to(device)
                        padding_mask_test = padding_mask_test.to(device)
                        y_parent_test = y_parent_test.to(device)
                        y_leaf_test = y_leaf_test.to(device)

                        # make the prediction
                        y_hat_parent, y_hat_leaf = model(x_test, padding_mask_test)

                        # Calculate the loss.
                        loss = loss_function(y_hat_parent, y_hat_leaf, y_parent_test, y_leaf_test)

                        testing_loss.append(loss.item())
                        
                        # collect data for confusion matrix
                        y_hat = torch.argmax(y_hat_leaf, dim=-1)
                        accuracy = (y_hat == y_leaf_test).float().mean().item()
                        batch_accuracies.append(accuracy)

                testing_loss = np.array(testing_loss).mean()
                print(f'Testing loss: {testing_loss:7.4f}')
                print(f'Testing accuracy for leaf classes: {np.array(batch_accuracies).mean():7.4f}')
                break
        print(f'Epoch: {epoch:03d} | '
              f'Mean training loss: {epoch_loss_training:7.4f} | '
              f'Mean validation loss {epoch_loss_validation:7.4f}')


if __name__ == '__main__':
    main()

# EOF
