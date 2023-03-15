class FloodsAnalysis:

    def __init__(self, hh_in_1_to_10=19, hh_in_1_to_100=855, hh_in_pmf=1959, 
                 avg_income_loss=818.9, cost_rebuild=473000, cost_repair=17000, 
                 discount_rate=0, prop_rebuild=0.5, prop_repair=0.5, 
                 weeks_rent_while_repairing=12, weeks_rent_while_rebuilding=1):
        # Number of households in the 1:10 flood zone
        self.hh_in_1_to_10 = hh_in_1_to_10
        # Number of households in the 1:100 flood zone
        self.hh_in_1_to_100 = hh_in_1_to_100
        # Number of households in the probable maximum flood zone
        self.hh_in_pmf = hh_in_pmf
        # Dictionary of house densities
        self.hh_dict = {10: self.hh_in_1_to_10, 100: self.hh_in_1_to_100, 1000: self.hh_in_pmf}
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
    
    def calculate_lost_income(self):
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
        repair = flood_analysis.calculate_repair_costs()[-1]
        rebuild = flood_analysis.calculate_rebuild_costs()[-1]
        income = flood_analysis.calculate_lost_income()[-1]
        rental = flood_analysis.calculate_rental_costs()[-1]


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


flood_analysis = FloodsAnalysis()

print(flood_analysis.calculate_repair_costs())
print(flood_analysis.calculate_rebuild_costs())
print(flood_analysis.calculate_lost_income())
print(flood_analysis.calculate_rental_costs())