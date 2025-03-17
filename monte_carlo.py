"""
Monte Carlo Portfolio Simulation: Age 22 vs. Age 32 Investment Strategy Comparison

This script simulates investment outcomes using historical S&P 500 returns to compare 
investment strategies starting at age 22 versus age 32.

Author: Yana Yerokhina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# Time period for historical data
START_DATE = '1970-01-01'
END_DATE = '2023-12-31'  # Using data until end of 2023

# Download S&P 500 historical data
print("Downloading S&P 500 historical data...")
ticker_data = yf.Ticker('^GSPC')
sp500_data = ticker_data.history(start=START_DATE, end=END_DATE, auto_adjust=False, actions=False, interval='1mo')

# Calculate monthly returns
sp500_data['Monthly_Return'] = sp500_data['Adj Close'].pct_change()
sp500_returns = sp500_data['Monthly_Return'].dropna()

# Function to run a single simulation
def simulate_portfolio(monthly_contributions: float, num_months: int, historical_returns: np.array) -> float:
    """
    Simulates the growth of an investment portfolio.
    
    Parameters:
    - monthly_contributions: Monthly investment amount.
    - num_months: Total number of months for the investment period.
    - historical_returns: Historical monthly returns of the S&P 500.
    
    Returns:
    - Final portfolio value after the investment period.
    """
    portfolio_value = 1000  # Initial investment
    
    for _ in range(num_months):
        monthly_return = np.random.choice(historical_returns)
        portfolio_value *= (1 + monthly_return)
        portfolio_value += monthly_contributions
    
    return portfolio_value

# Number of simulations
NUM_SIMULATIONS = 1000
MONTHLY_CONTRIBUTION = 1000

# Define investment scenarios
SCENARIOS = {
    "Start at 22": (65 - 22) * 12,  # 516 months
    "Start at 32": (65 - 32) * 12   # 396 months
}

# Run simulations
simulation_results = {}

for scenario, months in SCENARIOS.items():
    print(f"\nRunning {NUM_SIMULATIONS} simulations for {scenario}...")
    results = [simulate_portfolio(MONTHLY_CONTRIBUTION, months, sp500_returns.values) for _ in range(NUM_SIMULATIONS)]
    simulation_results[scenario] = np.array(results)

# Function to calculate confidence intervals
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean_val = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean_val, mean_val - h, mean_val + h

# Display results
print("\n--- Results Summary ---")
for scenario, results in simulation_results.items():
    mean_value = np.mean(results)
    std_dev = np.std(results)
    ci_95 = calculate_ci(results, 0.95)
    
    print(f"\n{scenario}")
    print(f"Mean portfolio value at retirement: ${mean_value:,.2f}")
    print(f"Standard deviation: ${std_dev:,.2f}")
    print(f"95% Confidence Interval: (${ci_95[1]:,.2f}, ${ci_95[2]:,.2f})")

# Plot results
plt.figure(figsize=(12, 8))
data = [simulation_results["Start at 22"], simulation_results["Start at 32"]]
labels = list(SCENARIOS.keys())

sns.boxplot(data=data, width=0.3)
plt.xticks([0, 1], labels)
plt.title('Retirement Portfolio Value Distribution: Age 22 vs. Age 32', fontsize=16)
plt.ylabel('Portfolio Value at Age 65 ($)', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Save results to a text file
with open('retirement_simulation_results.txt', 'w') as f:
    f.write("Monte Carlo Portfolio Simulation Results\n")
    f.write("======================================\n\n")
    f.write("Simulation Parameters:\n")
    f.write(f"- Initial investment: $1,000\n")
    f.write(f"- Monthly contribution: ${MONTHLY_CONTRIBUTION:,}\n")
    f.write(f"- Number of simulations: {NUM_SIMULATIONS:,}\n\n")
    for scenario, results in simulation_results.items():
        mean_value = np.mean(results)
        ci_95 = calculate_ci(results, 0.95)
        f.write(f"{scenario}\n")
        f.write(f"Mean Portfolio Value: ${mean_value:,.2f}\n")
        f.write(f"95% Confidence Interval: (${ci_95[1]:,.2f}, ${ci_95[2]:,.2f})\n\n")

print("\nSimulation complete. Results saved to 'retirement_simulation_results.txt'.")
