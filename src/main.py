class FloodsAnalysis:

    def __init__(self, hh_in_1_to_10, hh_in_1_to_100, hh_in_pmf, avg_income_loss, cost_rebuild, cost_repair, discount_rate, prop_rebuild, prop_repair, weeks_rent_while_repairing, weeks_rent_while_rebuilding):
        # Number of households in the 1:10 flood zone
        self.hh_in_1_to_10 = hh_in_1_to_10
        # Number of households in the 1:100 flood zone
        self.hh_in_1_to_100 = hh_in_1_to_100
        # Number of households in the probable maximum flood zone
        self.hh_in_pmf = hh_in_pmf
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

flood_analysis = FloodsAnalysis(100, 500, 1000, 5000, 20000, 10000, 5, 0.75, 0.25, 12, 24)

print(flood_analysis.hh_in_1_to_10)
print(flood_analysis.cost_rebuild)