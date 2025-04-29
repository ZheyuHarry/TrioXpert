"""
    We define the train function here
"""
import numpy as np
from tqdm import tqdm
from model.AutoEncoder import create_dataloader_AR
from torch import nn
import torch
import dgl
import torch.optim as optim
from utils.public_function import load_pkl
import time 

def train_autoencoder(model, train_loader, criterion, optimizer, scheduler, epochs=10, device='cpu', save_path='best_model.pth'):
    model = model.to(device)
    model.train()

    best_loss = float('inf')  

    # Early Stopping 
    PATIENCE = 20  
    early_stop_threshold = 1e-3 
    stop_count = 0  
    prev_loss = np.inf  

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            graphs = sample[1]
            graphs = graphs.to(device)
            inputs = sample[2].to(device).float()
            targets = sample[3].to(device).float()

            # Forward pass
            outputs = model(graphs, inputs)  
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate the average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        
        # Update the learning rate scheduler
        scheduler.step(avg_loss)
        
        # Print the loss for this epoch
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss}")

        if prev_loss - avg_loss < early_stop_threshold:
            stop_count += 1
            if stop_count == PATIENCE:
                print("Early stopping triggered.")
                break
        else:
            best_loss = avg_loss
            stop_count = 0
            prev_loss = avg_loss

            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch+1} with loss {best_loss:.6f}")

def train(model,train_path, batch_size = 32, num_nodes = 18, num_features = 613, learning_rate = 0.001, weight_decay = 1e-4, num_epochs = 20):
    train_samples = load_pkl(train_path)
    train_loader = create_dataloader_AR(train_samples, batch_size=batch_size,shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    # Initialize model, loss function, and optimizer
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Train the model
    start_time = time.time()
    train_autoencoder(model, train_loader, criterion, optimizer, scheduler, epochs=num_epochs, device=device, save_path="best_model.pth")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")