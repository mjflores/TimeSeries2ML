from modeloLSTM import ImprovedLSTM, train_lstm
from modeloMLP import basicMLP, train_basicMLP
from airPassengers import AirPassengersAnalyzer
import numpy as np
import matplotlib.pyplot as plt
import torch

def main():
    dir_TS = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    analyzer = AirPassengersAnalyzer()
    
    try:
        # Load and prepare data
        data = analyzer.read_data(dir_TS)
        analyzer.plot_time_series()
        print("Time series plot saved as 'air_passengers.png'")

        # Create sequences for time series
        seq_length = 10
        X, Y = analyzer.create_sequences(seq_length)
        print("Sequences created successfully")
        print("X:", X[0:5, :])
        print("Y:", Y[0:5, :])
        


    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
