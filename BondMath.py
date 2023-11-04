import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

rate = 0.05
coupon = 0.05
face_value = 10 ** 2
days_to_maturity = 10950
day_count_convention = 365
coupon_frequency = 2  
rate_changes = [0.03, 0.015, 0.005, -0.005, -0.015, -0.03]

class BondCalculator:

    def __init__(self, rate: float, coupon: float, face_value: float, days_to_maturity: int, day_count_convention: int, coupon_frequency: int):
        self.rate = rate
        self.coupon = coupon
        self.face_value = face_value
        self.days_to_maturity = days_to_maturity
        self.day_count_convention = day_count_convention
        self.coupon_frequency = coupon_frequency
    
    def calculate_bond_price(self, adjusted_rate: float) -> float:
        """
        Calculate the bond price for a given adjusted rate.
        """
        periods = self.days_to_maturity // (self.day_count_convention // self.coupon_frequency)
        period_coupon_payment = (self.coupon * self.face_value) / self.coupon_frequency
        cash_flows = np.full(periods, period_coupon_payment)
        cash_flows[-1] += self.face_value
        discount_factors = (1 + adjusted_rate / self.coupon_frequency) ** np.arange(1, periods + 1)
        present_value_flows = cash_flows / discount_factors
        return present_value_flows.sum()
    
    def calculate_bond_duration(self) -> float:
        """
        Calculate the bond's Macaulay duration and convert it to years.
        """
        periods = self.days_to_maturity // (self.day_count_convention // self.coupon_frequency)
        period_coupon_payment = (self.coupon * self.face_value) / self.coupon_frequency
        cash_flows = np.full(periods, period_coupon_payment)
        cash_flows[-1] += self.face_value
        discount_factors = (1 + self.rate / self.coupon_frequency) ** np.arange(1, periods + 1)
        present_value_flows = cash_flows / discount_factors
        weighted_times = np.arange(1, periods + 1) * present_value_flows
        macaulay_duration_periods = weighted_times.sum() / present_value_flows.sum()
        return macaulay_duration_periods / self.coupon_frequency  # Convert periods to years

    def calculate_bond_convexity(self) -> float:
        """
        Calculate the bond's convexity.
        """
        periods = self.days_to_maturity // (self.day_count_convention // self.coupon_frequency)
        period_coupon_payment = (self.coupon * self.face_value) / self.coupon_frequency
        cash_flows = np.full(periods, period_coupon_payment)
        cash_flows[-1] += self.face_value
        discount_factors = (1 + self.rate / self.coupon_frequency) ** np.arange(1, periods + 1)
        time_periods = np.arange(1, periods + 1)
        convexity = np.sum(cash_flows * time_periods * (time_periods + 1) / discount_factors**2)
        convexity /= (self.face_value * (1 + self.rate / self.coupon_frequency)**2) 
        return convexity / 100
    
    def calculate_modified_duration(self) -> float:
        """
        Calculate the bond's modified duration.
        """
        macaulay_duration = self.calculate_bond_duration()
        modified_duration = macaulay_duration / (1 + (self.rate / self.coupon_frequency))
        return modified_duration

# Bond calculator instance
bond_calculator = BondCalculator(rate, coupon, face_value, days_to_maturity, day_count_convention, coupon_frequency)

# Calculate base bond price and duration
base_price = bond_calculator.calculate_bond_price(rate)
base_duration = bond_calculator.calculate_bond_duration()

# Calculate bond prices with rate adjustments
data = [
    {
        'Rate Change': change,
        'Bond Price': (adjusted_price := bond_calculator.calculate_bond_price(rate + change)),
        'Estimated Price Using Duration': (estimated_price := base_price - (base_duration * base_price * change)),
        'Delta': adjusted_price - estimated_price
    }
    for change in rate_changes
]

# Create and format the DataFrame in one step
bond_dataframe = pd.DataFrame(data).set_index('Rate Change')
print("Bond Math:")
print(bond_dataframe)

# Define a range for coupons and yields
coupon_range = np.arange(0, 2 * rate + 0.01, 0.01)  # From 0% to 2x initial rate, in increments of 1%
yield_range = np.arange(0, 2 * rate + 0.01, 0.01)  # Same range for yields

# Function to calculate bond prices
def calculate_bond_values(func):
    # Use numpy to create a matrix of bond values
    values = np.array([[func(yield_rate, coupon) for yield_rate in yield_range] for coupon in coupon_range])
    return pd.DataFrame(values, index=coupon_range, columns=yield_range)

# Calculate bond prices
bond_prices_df = calculate_bond_values(lambda y, c: BondCalculator(y, c, face_value, days_to_maturity, day_count_convention, coupon_frequency).calculate_bond_price(y))

# Calculate bond convexities
bond_convexity_df = calculate_bond_values(lambda y, c: BondCalculator(y, c, face_value, days_to_maturity, day_count_convention, coupon_frequency).calculate_bond_convexity())

# Calculate Macaulay durations
bond_duration_df = calculate_bond_values(lambda y, c: BondCalculator(y, c, face_value, days_to_maturity, day_count_convention, coupon_frequency).calculate_bond_duration())

# Calculate modified durations
bond_modified_duration_df = calculate_bond_values(lambda y, c: BondCalculator(y, c, face_value, days_to_maturity, day_count_convention, coupon_frequency).calculate_modified_duration())

# Calculate duration differences
duration_difference_df = bond_duration_df - bond_modified_duration_df

# Calculate DV01s
dv01_df = bond_modified_duration_df * face_value * 0.0001

# Rename the index and columns for clarity and display the DataFrames
for df, name in zip([bond_prices_df, bond_convexity_df, bond_duration_df, bond_modified_duration_df, duration_difference_df, dv01_df],
                    ["Bond Price", "Bond Convexity", "Macaulay Duration", "Modified Duration", "Duration Difference", "DV01"]):
    df.index.name = 'Coupon Rate'
    df.columns.name = 'Yield Rate'
    print(name + ":")
    print(df)
    
# Function to plot bond price and convexity vs yield for each coupon rate
def plot_bond_metric(df, title, y_label):
    plt.figure(figsize=(10, 6))
    for coupon in df.columns:
        plt.plot(df.index, df[coupon], label=f'Coupon {coupon*100:.0f}%')

    plt.gca().invert_xaxis()
    plt.title(title)
    plt.xlabel('Yield')
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.show()

# 1. Plotting Bond Prices vs Yield for each Coupon Rate
plot_bond_metric(bond_prices_df, 'Bond Prices vs Yield for Different Coupons', 'Price')

# 2. Plotting Bond Convexity vs Yield for each Coupon Rate
plot_bond_metric(bond_convexity_df, 'Bond Convexity vs Yield for Different Coupons', 'Convexity')