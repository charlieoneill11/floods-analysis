import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class FloodsAnalysis:

    def __init__(self, hh_in_1_to_10=19, hh_in_1_to_100=855, hh_in_1_to_1000=1959, 
                 avg_income_loss=818.9, cost_rebuild=473000, cost_repair=17000, 
                 discount_rate=0, prop_rebuild=0.5, prop_repair=0.5, 
                 weeks_rent_while_repairing=12, weeks_rent_while_rebuilding=1):
        # Number of households in the 1:10 flood zone
        self.hh_in_1_to_10 = hh_in_1_to_10
        # Number of households in the 1:100 flood zone
        self.hh_in_1_to_100 = hh_in_1_to_100
        # Number of households in the probable maximum flood zone
        self.hh_in_1_to_1000 = hh_in_1_to_1000
        # Dictionary of house densities
        self.hh_dict = {10: self.hh_in_1_to_10, 100: self.hh_in_1_to_100, 1000: self.hh_in_1_to_1000}
        # Average income loss per household due to the flood
        self.avg_income_loss = avg_income_loss
        # Cost of rebuilding a household
        self.cost_rebuild = cost_rebuild
        # Cost of repairing a household
        self.cost_repair = cost_repair
        # Discount rate used to calculate the net present value of costs and benefits over time
        self.discount_rate = discount_rate
        # Proportion of households that choose to rebuild instead of moving
        self.prop_rebuild = prop_rebuild
        # Proportion of households that choose to repair instead of moving
        self.prop_repair = prop_repair
        # Number of weeks a household pays rent while repairing their home
        self.weeks_rent_while_repairing = weeks_rent_while_repairing
        # Number of weeks a household pays rent while rebuilding their home
        self.weeks_rent_while_rebuilding = weeks_rent_while_rebuilding

    def calculate_repair_costs(self):
        cc = CalculateCosts(self.hh_dict, [10, 100, 1000], self.cost_repair, self.prop_repair)
        return cc.calculate_costs()
    
    def calculate_rebuild_costs(self):
        cc = CalculateCosts(self.hh_dict, [10, 100, 1000], self.cost_rebuild, self.prop_rebuild)
        return cc.calculate_costs()
    
    def calculate_income_costs(self):
        cc = CalculateCosts(self.hh_dict, [10, 100, 1000], self.avg_income_loss, 1)
        return cc.calculate_costs()
    
    def calculate_rental_costs(self, weekly_rent=475):
        costs = []
        years = list(self.hh_dict.keys())
        for y in years:
            repair_rent = self.hh_dict[y]*self.prop_repair*self.weeks_rent_while_repairing*weekly_rent
            rebuild_rent = self.hh_dict[y]*self.prop_rebuild*self.weeks_rent_while_rebuilding*weekly_rent
            costs.append(repair_rent + rebuild_rent)
        # define probabilities
        probs = [1/years[0], 1/years[1], 1/years[2]]
        # calculate total
        total = probs[0]*costs[0] + ((probs[0]+probs[1])/2)*(costs[1]-costs[0]) + ((probs[1]+probs[2])/2)*(costs[2] - costs[1])
        return costs + [total]
    
    def calculate_total_costs(self):
        repair = self.calculate_repair_costs()
        rebuild = self.calculate_rebuild_costs()
        income = self.calculate_income_costs()
        rental = self.calculate_rental_costs()
        totals = []
        for i in range(len(repair)):
            totals.append((repair[i] + rebuild[i] + income[i] + rental[i]))
        return totals

    def cost_matrix(self):
        matrix = np.vstack([self.calculate_repair_costs(), 
                            self.calculate_rebuild_costs(), 
                            self.calculate_income_costs(),
                            self.calculate_rental_costs(),
                            self.calculate_total_costs()]).astype(int)
        return matrix

    def print_costs(self):
        matrix = self.cost_matrix()

        # define row and column labels
        rows = ['Repair', 'Rebuild', 'Income', 'Rental', 'Total']
        cols = ['1-in-10', '1-in-100', '1-in-1000', 'Annual exp. cost (all floods)']

        # create pandas data frame
        df = pd.DataFrame(data=matrix, index=rows, columns=cols)

        # show the data frame
        print(df)

    def expected_cost_to_hh(self):
        costs = []
        totals = self.calculate_total_costs()
        years = list(self.hh_dict.keys())
        probs = [1/y for y in years]
        for i, y in enumerate(years):
            c = totals[i]  # assuming the cost to HH is simply the total cost
            costs.append(c)
        total = probs[0]*costs[0] + ((probs[0]+probs[1])/2)*(costs[1]-costs[0]) + ((probs[1]+probs[2])/2)*(costs[2] - costs[1])
        return total

    def expected_cost_to_government(self):
        costs = []
        totals = self.calculate_total_costs()
        years = list(self.hh_dict.keys())
        probs = [1/y for y in years]
        for i, y in enumerate(years):
            c = 2.41 * totals[i]
            costs.append(c)
        total = probs[0]*costs[0] + ((probs[0]+probs[1])/2)*(costs[1]-costs[0]) + ((probs[1]+probs[2])/2)*(costs[2] - costs[1])
        return total


    def expected_cost_to_business(self):
        costs = []
        totals = self.calculate_total_costs()
        years = list(self.hh_dict.keys())
        probs = [1/y for y in years]
        for i, y in enumerate(years):
            c = 1.8689 * totals[i]
            costs.append(c)
        total = probs[0]*costs[0] + ((probs[0]+probs[1])/2)*(costs[1]-costs[0]) + ((probs[1]+probs[2])/2)*(costs[2] - costs[1])
        return total
    
class NPV(FloodsAnalysis):

    # This class inherits from the FloodsAnalysis class. 
    # It calculates the net present value of costs and benefits over time.
    # To do this, it uses a discount rate.

    def __init__(self, year_range=range(2022, 2043), discount_rate=0.0, 
                 hh_in_1_to_10=19, hh_in_1_to_100=855, hh_in_1_to_1000=1959):
        super().__init__(hh_in_1_to_10=hh_in_1_to_10, hh_in_1_to_100=hh_in_1_to_100, 
                         hh_in_1_to_1000=hh_in_1_to_1000)
        # Year range e.g. current year to 2042
        self.year_range = year_range
        self.discount_rate = discount_rate

    def retrieve_variables(self):
        # Retrieve variables from the FloodsAnalysis class
        matrix = self.cost_matrix()
        # Retrieve the final column from the cost matrix
        costs = matrix[:,-1]
        # Retrieve government and discount costs
        gov_cost = self.expected_cost_to_government()
        bus_cost = self.expected_cost_to_business()
        # Create a numpy vector of [costs, gov_cost, bus_cost]
        costs = np.append(costs, [gov_cost, bus_cost])
        # Turn costs (7x1) into a matrix (7xlen(year_range))
        costs = np.tile(costs, (len(self.year_range), 1))
        return costs
    
    def sum_costs(self):
        # Get the cost matrix using calculate_costs
        costs = self.retrieve_variables()
        # Sum each column and return the answer as a vector
        return np.sum(costs, axis=0)
    
    def npv(self, values):
        """
        Calculates the net present value of an investment by using a discount rate and a series of future payments 
        (negative values) and income (positive values).
        """
        periods = np.arange(1, len(values) + 1)
        discount_factors = 1 / ((1 + self.discount_rate) ** periods)
        npv = np.sum(values * discount_factors)
        return npv
    
    def npv_results(self):
        # Get the year-on-year cost matrix
        costs = self.retrieve_variables()
        # Get household, government and business costs as the 5th, 6th and 7th columns
        hh_costs = costs[:, 4]
        gov_costs = costs[:, 5]
        bus_costs = costs[:, 6]
        # Calculate the net present value of the costs
        hh_npv = self.npv(hh_costs)
        gov_npv = self.npv(gov_costs)
        bus_npv = self.npv(bus_costs)
        return hh_npv, gov_npv, bus_npv
    
    def npv_table(self, discount_rates=[0, 2, 5]):
        # Calculate the NPVs for each discount rate
        original_dr = self.discount_rate
        npvs = []
        for rate in discount_rates:
            self.discount_rate = rate
            hh_npv, gov_npv, bus_npv = self.npv_results()
            total_npv = hh_npv + gov_npv + bus_npv
            npvs.append([hh_npv / 1e6, gov_npv / 1e6, bus_npv / 1e6, total_npv / 1e6])

        # Create a dataframe with the results
        df = pd.DataFrame(npvs, columns=['HH NPV $m', 'Gov NPV $m', 'Business NPV $m', 'Total $m'],
                        index=[f'Discount rate {rate}%' for rate in discount_rates])

        # Format the dataframe to show results in units of million dollars
        df['HH NPV $m'] = df['HH NPV $m'].map('${:,.2f}'.format)
        df['Gov NPV $m'] = df['Gov NPV $m'].map('${:,.2f}'.format)
        df['Business NPV $m'] = df['Business NPV $m'].map('${:,.2f}'.format)
        df['Total $m'] = df['Total $m'].map('${:,.2f}'.format)

        # Reset discount rate
        self.discount_rate = original_dr

        return df


        

class CalculateCosts:
    
    def __init__(self, hh_dict, years, cost, proportion):
         self.hh_dict = hh_dict
         self.years = years
         self.cost = cost 
         self.proportion = proportion

    def calculate_costs(self):
        costs = []
        for y in self.years:
            costs.append(self.hh_dict[y]*self.cost*self.proportion)
        # define probabilities
        probs = [1/self.years[0], 1/self.years[1], 1/self.years[2]]
        # calculate total
        total = probs[0]*costs[0] + ((probs[0]+probs[1])/2)*(costs[1]-costs[0]) + ((probs[1]+probs[2])/2)*(costs[2] - costs[1])
        return costs + [total]
    

def logarithmic_func(x, a, b):
    return a * np.log(x) + b

def fit_logarithmic_curve(cost_points, probability_points):
    popt, pcov = curve_fit(logarithmic_func, cost_points, probability_points)
    return popt


def generate_plot(hh_in_1_to_10, hh_in_1_to_100, hh_in_1_to_1000):
    # Calculate the costs
    flood_analysis = FloodsAnalysis(hh_in_1_to_10, hh_in_1_to_100, hh_in_1_to_1000)
    repair_costs = flood_analysis.calculate_repair_costs()
    rebuild_costs = flood_analysis.calculate_rebuild_costs()
    income_costs = flood_analysis.calculate_income_costs()
    rental_costs = flood_analysis.calculate_rental_costs()

    # Calculate expected annual costs
    cost_matrix = np.vstack([repair_costs, rebuild_costs, income_costs, rental_costs])
    expected_annual_costs = [(1/y)*cost_matrix[:, i].sum() for i, y in enumerate([10, 100, 1000])]

    # Fit a logarithmic function to the data
    a, b = fit_logarithmic_curve(expected_annual_costs, [0.1, 0.01, 0.001])

    # Generate the logarithmic curves
    x_vals = np.linspace(min(expected_annual_costs), max(expected_annual_costs), 100)
    y_vals = logarithmic_func(x_vals, a, b)

    # Create the figures
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))

    # Plot the one-off costs
    ax[0].plot(expected_annual_costs, [0.1, 0.01, 0.001], 'o', label='Data Points')
    ax[0].plot(x_vals, y_vals, label=r'Logarithmic Fit: $y = a\ln(x) + b$')
    ax[0].legend(labels=['Data Points', fr'$y = {a:.2f}\ln(x) + {b:.2f}$'])
    ax[0].set_xlabel('Cost ($M)')
    ax[0].set_ylabel('Probability')
    ax[0].set_title('One-off cost of different flood levels')

    # Plot the expected annual costs
    ax[1].plot(expected_annual_costs, [0.1, 0.01, 0.001], 'o', label='Data Points')
    ax[1].plot(x_vals, y_vals, label=r'Logarithmic Fit: $y = a\ln(x) + b$')
    ax[1].legend(labels=['Data Points', fr'$y = {a:.2f}\ln(x) + {b:.2f}$'])
    ax[1].set_xlabel('Cost ($M)')
    ax[1].set_ylabel('Probability')
    ax[1].set_title('Expected annual cost of different flood levels')

    # Save the plot as a PNG
    fig.savefig('static/plot.png', bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(fig)

def generate_new_plots(hh_in_1_to_10, hh_in_1_to_100, hh_in_1_to_1000):
    # Create the contour plot for hh_in_1_to_10 and hh_in_1_to_100
    hh_in_1_to_10_range = np.linspace(5, 1000, 100)
    hh_in_1_to_100_range = np.linspace(500, 5000, 100)
    hh_in_1_to_10_grid, hh_in_1_to_100_grid = np.meshgrid(hh_in_1_to_10_range, hh_in_1_to_100_range)
    expected_cost_grid = np.zeros_like(hh_in_1_to_10_grid)

    for i in range(len(hh_in_1_to_10_range)):
        for j in range(len(hh_in_1_to_100_range)):
            flood_analysis = FloodsAnalysis(hh_in_1_to_10_range[i], hh_in_1_to_100_range[j], hh_in_1_to_1000)
            expected_cost = flood_analysis.calculate_total_costs()[-1]
            expected_cost_grid[j, i] = expected_cost

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot on first axis
    contour = ax1.contourf(hh_in_1_to_10_grid, hh_in_1_to_100_grid, expected_cost_grid, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax1)
    ax1.set_xlabel('Homes in 1-in-10 flood range')
    ax1.set_ylabel('Homes in 1-in-100 flood range')
    ax1.set_title('Annual expected cost ($M)')

    # Create the contour plot for hh_in_1_to_100 and hh_in_1_to_1000
    hh_in_1_to_100_range = np.linspace(5, 1000, 100)
    hh_in_1_to_1000_range = np.linspace(1500, 15000, 100)
    hh_in_1_to_100_grid, hh_in_1_to_1000_grid = np.meshgrid(hh_in_1_to_100_range, hh_in_1_to_1000_range)
    expected_cost_grid = np.zeros_like(hh_in_1_to_100_grid)

    for i in range(len(hh_in_1_to_100_range)):
        for j in range(len(hh_in_1_to_1000_range)):
            flood_analysis = FloodsAnalysis(hh_in_1_to_10, hh_in_1_to_100_range[i], hh_in_1_to_1000_range[j])
            expected_cost = flood_analysis.calculate_total_costs()[-1]
            expected_cost_grid[j, i] = expected_cost

    # Plot on second axis
    contour = ax2.contourf(hh_in_1_to_100_grid, hh_in_1_to_1000_grid, expected_cost_grid, cmap='viridis')
    cbar = plt.colorbar(contour, ax=ax2)
    ax2.set_xlabel('Homes in 1-in-100 flood range')
    ax2.set_ylabel('Homes in 1-in-1000 flood range')
    ax2.set_title('Annual expected cost ($M)')

    # Save figure
    fig.tight_layout()
    fig.savefig('static/plot3.png')
    plt.close(fig)

# npv = NPV(year_range=range(2022, 2043), discount_rate=0.02)
# print(npv.npv_table())
# fa = FloodsAnalysis()
# print(fa.expected_cost_to_hh())
# print(fa.expected_cost_to_government())
# print(fa.expected_cost_to_business())