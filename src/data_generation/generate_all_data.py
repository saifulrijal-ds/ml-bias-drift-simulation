import os
import yaml
import pandas as pd
from data_generator import IndonesianLoanDataGenerator
from bias_injector import BiasInjector
from drift_simulator import DataDriftSimulator

def load_params():
    """Load parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params['data_generation']

def generate_all_datasets(params):
    """Generate all datasets needed for the simulation using parameters from params.yaml."""
    print("===ML Bias & Drift Simulation in Finance===")
    print("Generating all datasets for the simulation...")

    # Create data directories - we'll create these but won't use them directly in save calls
    # The generator classes already have these paths hardcoded
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('data/synthetic/biased', exist_ok=True)
    os.makedirs('data/synthetic/time_series', exist_ok=True)
    os.makedirs('data/synthetic/time_series/quarterly', exist_ok=True)
    os.makedirs('data/synthetic/time_series/monthly', exist_ok=True)

    # 1. Generate baseline unbiased dataset
    print("\n1. Generating baseline dataset...")
    generator = IndonesianLoanDataGenerator()
    baseline_data = generator.generate_dataset(n_samples=params['baseline_samples'])
    
    # Simply use the filename - the path is handled by the generator class
    baseline_path = generator.save_dataset(baseline_data, "baseline_loan_data.csv")

    # 2. Generate datasets with individual bias types
    print("\n2. Generating datasets with individual bias types...")
    injector = BiasInjector()
    
    bias_types = params['bias_types']
    bias_severities = params['bias_severities']

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
            
            # Let the injector use its default path
            injector.save_biased_dataset(biased_data, bias_type, severity, output_dir='data/synthetic/biased')

    # 3. Generate datasets with multiple bias types
    print("\n3. Generating datasets with multiple bias types")
    bias_combinations = params['bias_combinations']
    
    for combo in bias_combinations:
        severity = 1.0  # Using fixed severity for combinations
        print(f"  - Generating dataset with combined biases: {', '.join(combo)} (severity: {severity})...")
        multi_biased = injector.inject_multiple_biases(baseline_data, biases=combo, severity=severity)
        
        # Let the injector use its default path
        injector.save_biased_dataset(multi_biased, combo, severity, output_dir='data/synthetic/biased')

    # 4. Generate time series data with drift
    print("\n4. Generating time series data with drift...")

    # Use the most biased dataset as base for drift simulation
    full_biased = injector.inject_multiple_biases(
        baseline_data,
        biases=['geographic', 'gender', 'occupation', 'digital_access'],
        severity=1.0
    )

    # Create drift simulator
    drift_simulator = DataDriftSimulator()

    # Generate quarterly data
    print("  - Generating quarterly data...")
    quarterly_params = params['quarterly']
    time_series_quarterly = drift_simulator.generate_time_series_data(
        full_biased,
        start_date=quarterly_params['start_date'],
        end_date=quarterly_params['end_date'],
        interval_days=quarterly_params['interval_days'],
        samples_per_period=quarterly_params['samples_per_period']
    )
    
    # Use the default path for quarterly data with a subdirectory name
    quarterly_path = drift_simulator.save_time_series_data(
        time_series_quarterly,
        output_dir='data/synthetic/time_series/quarterly'
    )
    
    # Generate monthly data
    print("  - Generating monthly data...")
    monthly_params = params['monthly']
    time_series_monthly = drift_simulator.generate_time_series_data(
        full_biased,
        start_date=monthly_params['start_date'],
        end_date=monthly_params['end_date'],
        interval_days=monthly_params['interval_days'],
        samples_per_period=monthly_params['samples_per_period']
    )
    
    # Use the default path for monthly data with a subdirectory name
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
    params = load_params()
    generate_all_datasets(params)