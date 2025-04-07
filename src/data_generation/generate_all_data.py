import os
import pandas as pd
from data_generator import IndonesianLoanDataGenerator
from bias_injector import BiasInjector
from drift_simulator import DataDriftSimulator

def generate_all_datasets():
    """Generate all datasets needed for the simulation."""
    print("===ML Bias & Drift Simulation in Finance===")
    print("Generating all datasets for the simulation...")

    # Create data directories
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('data/synthetic/biased', exist_ok=True)
    os.makedirs('data/synthetic/time_series', exist_ok=True)

    # 1. Generate baseline unbiased dataset
    print("\n1. Generating baseline dataset...")
    generator = IndonesianLoanDataGenerator()
    baseline_data = generator.generate_dataset(n_samples=50000)
    baseline_path = generator.save_dataset(baseline_data, "baseline_loan_data.csv")

    # 2. Generate datasets with individual bias types
    print("\n2. Generating datasets with individual bias types...")
    injector = BiasInjector()
    
    bias_types = ['geographic', 'gender', 'digital_access', 'occupation']
    bias_severities = [0.5, 1.0, 1.5]

    for bias_type in bias_types:
        for severity in bias_severities:
            print(f"  - Generating dataset with {bias_type} bias (severity: {severity})...")
            if bias_type == 'geographic':
                biased_data = injector.inject_geographic_bias(baseline_data, severity)
            elif bias_type == 'gender':
                biased_data = injector.inject_gender_bias(baseline_data, severity)
            elif bias_type == 'digital_access':
                biased_data = injector.inject_digital_access_bias(baseline_data, severity)
            elif bias_type == 'occupation':
                biased_data = injector.inject_occupation_bias(baseline_data, severity)
            
            injector.save_biased_dataset(biased_data, bias_type, severity, 'data/synthetic/biased')

        
        # 3. Generate datasets with multiple bias types
        print("\n3. Generating datasets with multiple bias types")
        bias_combinations = [
        ['geographic', 'gender'],
        ['digital_access', 'occupation'],
        ['geographic', 'occupation'],
        ['gender', 'digital_access'],
        ['geographic', 'gender', 'occupation', 'digital_access']
    ]
    
    for combo in bias_combinations:
        severity = 1.0
        print(f"  - Generating dataset with combined biases: {', '.join(combo)} (severity: {severity})...")
        multi_biased = injector.inject_multiple_biases(baseline_data, biases=combo, severity=severity)
        injector.save_biased_dataset(multi_biased, combo, severity, 'data/synthetic/biased')

    # 4. Generate time series data with drift
    print("\n4. Generating time series data with drift...")

    # Use the most biased dataset as base for drift simulation
    full_biased = injector.inject_multiple_biases(
        baseline_data,
        biases=['geographic', 'gender', 'occupation', 'digital_access'],
        severity=1.0
    )

    # create drift simulator
    drift_simulator = DataDriftSimulator()

    # Generate quarterly data for 3 years
    print("  - Generating quarterly data for 3 years...")
    time_series_quarterly = drift_simulator.generate_time_series_data(
        full_biased,
        start_date='2021-01-01',
        end_date='2023-12-31',
        interval_days=90,  # Quarterly
        samples_per_period=5000
    )
    
    quarterly_path = drift_simulator.save_time_series_data(
        time_series_quarterly,
        output_dir='data/synthetic/time_series/quarterly'
    )
    
    # Generate monthly data for 1 year (more granular)
    print("  - Generating monthly data for 1 year...")
    time_series_monthly = drift_simulator.generate_time_series_data(
        full_biased,
        start_date='2022-01-01',
        end_date='2022-12-31',
        interval_days=30,  # Monthly
        samples_per_period=2000
    )
    
    monthly_path = drift_simulator.save_time_series_data(
        time_series_monthly,
        output_dir='data/synthetic/time_series/monthly'
    )

     # 5. Summary
    print("\n=== Data Generation Complete ===")
    print(f"Baseline dataset: {baseline_path}")
    print(f"Biased datasets: {len(bias_types) * len(bias_severities) + len(bias_combinations)} files in data/synthetic/biased/")
    print(f"Time series data (quarterly): {len(time_series_quarterly)} periods in data/synthetic/time_series/quarterly/")
    print(f"Time series data (monthly): {len(time_series_monthly)} periods in data/synthetic/time_series/monthly/")

if __name__ == "__main__":
    generate_all_datasets()