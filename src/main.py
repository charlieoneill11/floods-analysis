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
    flood_analysis = FloodsAnalysis(hh_in_1_to_10=hh_in_1_to_10,
                                    hh_in_1_to_100=hh_in_1_to_100,
                                    hh_in_1_to_1000=hh_in_1_to_1000)
    cost_points = flood_analysis.calculate_total_costs()[:-1]
    probability_points = [0.1, 0.01, 0.001]

    # Fit logarithmic curve to data
    a, b = fit_logarithmic_curve(cost_points, probability_points)

    # Create a range of cost values to generate the curve
    x_vals = np.linspace(min(cost_points), max(cost_points), 100)

    # Generate the logarithmic curve using the fitted parameters
    y_vals = logarithmic_func(x_vals, a, b)

    # Plot the data points and the logarithmic curve
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(cost_points, probability_points, 'o', label='Data Points')
    ax.plot(x_vals, y_vals, label=r'Logarithmic Fit: $y = a\ln(x) + b$')

    # Format the legend to include the equation for the logarithmic curve
    eqn_str = fr'$y = {a:.2f}\ln(x) + {b:.2f}$'
    ax.legend(labels=['Data Points', eqn_str])

    # Set the axis labels and title
    ax.set_xlabel('Cost ($M)')
    ax.set_ylabel('Probability')
    ax.set_title('One-off cost of different flood levels')

    plt.subplots_adjust(bottom=0.2)

    # Save the plot as an image file
    plt.savefig('static/plot.png', dpi=200)

npv = NPV(year_range=range(2022, 2043), discount_rate=0.02)
print(npv.npv_table())
# print(fa.expected_cost_to_government())
# print(fa.expected_cost_to_business())