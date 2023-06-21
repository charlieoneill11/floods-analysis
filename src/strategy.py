import numpy as np
import pandas as pd
from typing import Callable
from main import FloodsAnalysis

class Strategy:
    def __init__(self, years: range):
        self.years = years

    def calculate_strategy(self, strategy_file: str, base_cost_func: Callable[[int], tuple]):
        strategy_data = np.loadtxt(strategy_file)
        data = []

        for i, year in enumerate(self.years):
            num_hh_bought, num_hh_left_1_10, num_hh_left_1_100, num_hh_left_1_1000 = strategy_data[i]
            annual_hh_cost_base, annual_gov_cost_base, annual_bus_cost_base = base_cost_func(year)

            # TODO: Use the numbers and the base costs to calculate the costs for this strategy.
            # Add these costs to the `data` list in the appropriate format.

        return pd.DataFrame(data)
    
    @staticmethod
    def base_cost_func(self, year: int) -> tuple:
        floods_analysis = FloodsAnalysis()
        
        annual_hh_cost_base = floods_analysis.expected_cost_to_hh()
        annual_gov_cost_base = floods_analysis.expected_cost_to_government()
        annual_bus_cost_base = floods_analysis.expected_cost_to_business()

        return annual_hh_cost_base, annual_gov_cost_base, annual_bus_cost_base


    def calculate_strategies(self, strategies: list, base_cost_func: Callable[[int], tuple]):
        strategy_dfs = []

        for strategy_file in strategies:
            df = self.calculate_strategy(strategy_file, base_cost_func)
            strategy_dfs.append(df)

        return strategy_dfs
    
if __name__ == "__main__":
    strategy_comp_buyback = Strategy("compulsory_buyback")
    strategy_comp_buyback.run()

    strategy_vol_buyback = Strategy("voluntary_buyback")
    strategy_vol_buyback.run()

    strategy_vol_landswap = Strategy("voluntary_landswap")
    strategy_vol_landswap.run()

    strategy_comp_landswap = Strategy("compulsory_landswap")
    strategy_comp_landswap.run()
    