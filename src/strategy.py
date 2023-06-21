import numpy as np
import pandas as pd
from typing import Callable
from main import FloodsAnalysis
import re

class Strategy:
    def __init__(self, years: range):
        self.years = years

    def calculate_strategy(self, strategy_file: str):
        # Load the strategy data from the file
        strategy_data = np.loadtxt(strategy_file)

        # Extract annual costs for households, government, and businesses
        annual_hh_cost = strategy_data[:, -1]
        annual_gov_cost = 2.41 * annual_hh_cost  # Gov vector is 2.41 times the HH vector
        annual_bus_cost = 1.87 * annual_hh_cost  # Business vector is 1.87 times the HH vector

        # Stack these arrays together
        data = np.vstack([annual_hh_cost, annual_gov_cost, annual_bus_cost]).T

        # Calculate the cost without program
        base_costs = np.array([self.base_cost_func() for _ in self.years])
        # Add this to data, each as its own column
        data = np.concatenate((data, base_costs), axis=1)

        # Calculate the difference between cost with and without program, for each sector
        hh_diff = data[:, 0] - data[:, 3]
        gov_diff = data[:, 1] - data[:, 4]
        bus_diff = data[:, 2] - data[:, 5]
        # Add this to data, each as its own column
        data = np.concatenate((data, hh_diff.reshape(-1, 1), gov_diff.reshape(-1, 1), bus_diff.reshape(-1, 1)), axis=1)

        # Calculate the cost of the program
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
    
    def npv(self, cashflows, discount_rate):
        years = np.arange(len(cashflows))
        discounted_cashflows = cashflows / (1 + discount_rate)**years
        return discounted_cashflows.sum()
    
    def summarise_strategy(self, df):
        summary_data = []
        rate_strings = []
        for rate in [0, 0.02]:
            rate_str = f"{int(rate * 100)}%"
            rate_strings.append(rate_str)
            row_data = {}
            for entity in ['HH', 'gov', 'bus']:
                npv_with_program = self.npv(df[f'{entity}_cost_with_program'], rate)
                npv_without_program = self.npv(df[f'{entity}_cost_without_program'], rate)
                npv_difference = self.npv(df[f'{entity}_cost_difference'], rate)
                row_data.update({
                    f'NPV {entity}': npv_with_program,
                    f'NPV {entity} base': npv_without_program,
                    f'Difference in NPV {entity}': npv_difference
                })
            npv_program = self.npv(df['program_cost'], rate)
            row_data[f'NPV GOV program'] = npv_program
            summary_data.append(row_data)
        summary_df = pd.DataFrame(summary_data, index=rate_strings)
        return summary_df
    
    def final_summary(self, strategies: list, discount_rates=[0, 0.02]):
        summary_data = []
        for rate in discount_rates:
            rate_str = f"{int(rate * 100)}% Discount rate"
            row_data = {}
            for strategy_file in strategies:
                df = self.calculate_strategy(strategy_file)
  
                # Extract the file name from the path
                file_name = strategy_file.split('/')[-1]
                # Remove the file extension
                file_name_without_extension = file_name.split('.')[0]
                # Convert the file name to title case
                strategy_name = file_name_without_extension.replace("_", " ").title()

                row_data.update({
                    f'Total ({strategy_name})': self.npv(df['HH_cost_difference'] + df['gov_cost_difference'] + df['bus_cost_difference'], rate),
                    f'Avoided HH cost ({strategy_name})': self.npv(df['HH_cost_without_program'] - df['HH_cost_with_program'], rate),
                    f'Avoided Gov cost ({strategy_name})': self.npv(df['gov_cost_without_program'] - df['gov_cost_with_program'], rate),
                    f'Avoided Bus cost ({strategy_name})': self.npv(df['bus_cost_without_program'] - df['bus_cost_with_program'], rate)
                })
            summary_data.append(row_data)
        summary_df = pd.DataFrame(summary_data, index=[f"{int(rate * 100)}% Discount rate" for rate in discount_rates])
        return summary_df
        
if __name__ == "__main__":
    years = range(2022, 2043)  # define the range of years
    strategies = ["../data/compulsory_buyback.txt", "../data/voluntary_buyback.txt", "../data/voluntary_landswap.txt"]
    
    strat = Strategy(years)
    strategy_dfs = strat.calculate_strategies(strategies)
    for df in strategy_dfs:
        print(df)