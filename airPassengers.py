import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Union


class AirPassengersAnalyzer:
    def __init__(self):
        self.data = None
    
    
    def read_data(self, filename='AirPassengers.csv'):
        """Read the AirPassengers dataset"""
        self.data = pd.read_csv(filename)
        self.data['Date'] = pd.to_datetime(self.data['Month'])
        self.data.set_index('Date', inplace=True)
        return self.data
    
    def plot_time_series(self, title="Air Passengers Over Time"):
        """Plot the time series data"""
        if self.data is None:
            raise ValueError("Data not loaded. Call read_data() first.")
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Passengers'])
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Number of Passengers')
        plt.grid(True)
        plt.savefig('plots/air_passengers.png')
        plt.close()
    
    def create_sequences(self, p):
        """
        Create X and Y matrices for time series analysis
        Args:
            data: time series data (array or list)
            p: number of lags
        Returns:
            X: matrix with columns x_{t-p},...,x(t)
            Y: matrix with column x_{t+1}
        """
        #data = self.data.index
        #print(data.shape)
        data = self.data['Passengers'].values
        n = len(data)
        X, Y = None, None
        # Validate inputs
        if p <= 0:
            raise ValueError("Lag p must be non-negative or cero")
        if p >= n-1:
            raise ValueError(f"Lag p must be less than {n-1} for this dataset")
        if n < p + 2:
            raise ValueError("Not enough data points to create sequences")
        else:
            # Create matrices
            X = np.zeros((n-p-1, p+1))  # Each row has p+1 values [x(t-p), ..., x(t)]
            Y = np.zeros((n-p-1, 1))    # Each row has the next value x(t+1)
            
            # Fill matrices
            for i in range(n-p-1):
                # X[i] contains values from (i) to (i+p)
                X[i,:] = data[i:(i+p+1)]
                # Y[i] contains the next value after the sequence
                Y[i,0] = data[i+p+1]


            plt.imshow(X, aspect='auto', cmap='hot')
            plt.colorbar()
            plt.title(f'X Matrix {X.shape}')
            plt.xlabel('Time Steps')
            plt.ylabel('Features')
            plt.savefig('plots/X_matrix.png')
            plt.close()
            
            plt.subplot(1, 2, 1)
            plt.scatter(X[:,0], Y, alpha=0.5)
            plt.title(f'Scatter plot of X[:,0] vs Y')
            plt.xlabel('X[:,0]')
            plt.ylabel('Y')

            plt.subplot(1, 2, 2)
            plt.scatter(X[:,0], X[:,1], alpha=0.5)
            plt.title(f'Scatter plot of X[:,0] vs X[:,1]')
            plt.xlabel('X[:,0]')
            plt.ylabel('X[:,1]')

            plt.tight_layout()
            plt.savefig('plots/scatter_plot.png')
            plt.close()

            return X, Y





def prepare_data(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2, 
                     batch_size: int = 16) -> Tuple[DataLoader, DataLoader, MinMaxScaler]:
    """
    Prepare data for LSTM model training and evaluation.
    
    Args:
        X: Input features (n_samples, seq_len, n_features)
        Y: Target values (n_samples, 1)
        test_size: Fraction of data to use for testing
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_loader, test_loader, y_scaler)
    """
    # Initialize scalers
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Scale features (X)
    # For 3D input (samples, timesteps, features)
    # We need to scale each feature across all samples and timesteps
    n_samples, seq_len, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)  # (samples * timesteps, features)
    X_scaled = x_scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)
    
    # Scale target (Y)
    Y_scaled = y_scaler.fit_transform(Y.reshape(-1, 1))
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    Y_tensor = torch.FloatTensor(Y_scaled)
    
    # Split into train and test sets
    train_size = int((1 - test_size) * len(X_tensor))
    
    X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
    Y_train, Y_test = Y_tensor[:train_size], Y_tensor[train_size:]
    
    # Create datasets
    train_data = TensorDataset(X_train, Y_train)
    test_data  = TensorDataset(X_test, Y_test)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, x_scaler, y_scaler