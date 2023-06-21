import numpy as np
import pandas as pd
from typing import Callable
from main import FloodsAnalysis

class Strategy:
    def __init__(self, years: range):
        self.years = years

    def calculate_strategy(self, strategy_file: str):
        data = []

        for i, year in enumerate(self.years):

            # Calculate the cost with program
            annual_hh_cost = np.array([13.69, 13.25, 12.53, 11.81, 11.08, 10.36, 
                                    9.64, 8.91, 8.19, 7.47, 6.74, 6.02, 5.3, 4.58, 
                                    3.85, 3.13, 2.41, 1.6, 1.57, 1.55, 1.54])[i]
            annual_gov_cost = 2.41 * annual_hh_cost  # Gov vector is 2.41 times the HH vector
            annual_bus_cost = 1.87 * annual_hh_cost  # Business vector is 1.87 times the HH vector

            data.append([annual_hh_cost, annual_gov_cost, annual_bus_cost])

        # Calculate the cost without program
        base_costs = np.array([self.base_cost_func() for year in self.years])
        # Add this to data, each as its own column
        data = np.concatenate((data, base_costs), axis=1)

        # Calculate the difference between cost with and without program, for each sector
        hh_diff = data[:, 0] - data[:, 3]
        gov_diff = data[:, 1] - data[:, 4]
        bus_diff = data[:, 2] - data[:, 5]
        # Add this to data, each as its own column
        data = np.concatenate((data, hh_diff.reshape(-1, 1), gov_diff.reshape(-1, 1), bus_diff.reshape(-1, 1)), axis=1)

        # Calculate the cost of the program
        strategy_data = np.loadtxt(strategy_file)
        program_cost = 0.52 * strategy_data[:, 0]
        # Add this as a column to data
        data = np.concatenate((data, program_cost.reshape(-1, 1)), axis=1)

        return pd.DataFrame(data, columns=["HH_cost_with_program", "gov_cost_with_program", "bus_cost_with_program", 
                                           "HH_cost_without_program", "gov_cost_without_program", "bus_cost_without_program", 
                                           "HH_cost_difference", "gov_cost_difference", "bus_cost_difference", "program_cost"], index=self.years)

    def base_cost_func(self):
        floods_analysis = FloodsAnalysis()
        
        annual_hh_cost_base = floods_analysis.expected_cost_to_hh()
        annual_gov_cost_base = floods_analysis.expected_cost_to_government()
        annual_bus_cost_base = floods_analysis.expected_cost_to_business()

        return annual_hh_cost_base, annual_gov_cost_base, annual_bus_cost_base

    def calculate_strategies(self, strategies: list):
        strategy_dfs = []

        for strategy_file in strategies:
            df = self.calculate_strategy(strategy_file)
            strategy_dfs.append(df)

        return strategy_dfs
    
if __name__ == "__main__":
    years = range(2022, 2043)  # define the range of years
    strategies = ["../data/compulsory_buyback.txt", "../data/voluntary_buyback.txt", "../data/voluntary_landswap.txt"]
    
    strategy = Strategy(years)
    strategy_dfs = strategy.calculate_strategies(strategies)

    # For now, we'll just print the DataFrames to the console
    for df in strategy_dfs:
        print(df)