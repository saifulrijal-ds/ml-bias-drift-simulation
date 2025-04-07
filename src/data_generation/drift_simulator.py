"""
Data Drift Simulator for Indonesian Loan Applications

This module creates various types of data drift scenarios based on 
economic, seasonal, regulatory, and market changes in the Indonesian 
financial context.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class DataDriftSimulator:
    """
    Generates data drift scenarios in Indonesian loan application data,
    simulating changes over time in the financial landscape.
    """

    def __init__(self, random_seed=42):
        """Initialize the drift simulator with a random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def apply_economic_drift(self, df, time_period, inflation_rate=0.05):
        """
        Apply economic drift effects like inflation to the data.

        Args:
            df: DataFrame containing loan application data
            time_period: String date representing the period for which drift is applied
            inflation_rate: Annual inflation rate (default 5%)
            
        Returns:
            DataFrame with economic drift effects
        """
        df = df.copy()

        # Parse dates
        current_date = datetime.strptime(time_period, '%Y-%m-%d')
        reference_date = datetime.strptime('2021-01-01', '%Y-%m-%d')

        # Calculate time difference in years
        years_diff = (current_date - reference_date).days / 365.25

        # Calculate cumulative inflation effect
        inflation_factor = (1 + inflation_rate) ** years_diff

        # Apply inflation to income loan amounts
        # Income increases with inflation
        df['Monthly_Income'] = df['Monthly_Income'] * inflation_factor

        # Loan amounts also increase, but not at exactly the same rate
        # to create a drift in the relationship between income and loan amount
        loan_factor = inflation_factor * (1 + random.uniform(-0.02, 0.02))
        df['Loan_Amount'] = df['Loan_Amount'] * loan_factor

        # Existing debt also affected by inflation
        df['Existing_Debt_Amount'] = df['Existing_Debt_Amount'] * inflation_factor

        # Recalculate payment to income ratio
        df['Monthly_Payment'] = df['Loan_Amount'] / df['Loan_Term'] * 1.1
        df['Payment_to_Income_Ratio'] = df['Monthly_Payment'] / df['Monthly_Income']

        return df
    
    def apply_seasonal_effects(self, df, date):
        """
        Apply seasonal effects like Ramadan, year-end bonuses.

        Args:
            df: DataFrame containing loan application data
            date: Date string for which seasonal effects should be applied
            
        Returns:
            DataFrame with seasonal effects
        """
        df = df.copy()

        # Convert to datetime
        current_date = datetime.strptime(date, '%Y-%m-%d')
        month = current_date.month

        # Ramadan dates (approximate for simulation purposes)
        ramadan_periods = {
            2021: {'start': '2021-04-12', 'end': '2021-05-12'},
            2022: {'start': '2022-04-02', 'end': '2022-05-02'},
            2023: {'start': '2023-03-22', 'end': '2023-04-21'},
            2024: {'start': '2024-03-10', 'end': '2024-04-09'}
        }

        # Year-end bonus periods (November-December)
        is_year_end = month in [11, 12]

        # Check if current date is during Ramadan
        is_ramadan = False
        year = current_date.year
        if year in ramadan_periods:
            ramadan_start = datetime.strptime(ramadan_periods[year]['start'], '%Y-%m-%d')
            ramadan_end = datetime.strptime(ramadan_periods[year]['end'], '%Y-%m-%d')
            is_ramadan = ramadan_start <= current_date <= ramadan_end

        # Apply Ramadan effects - increase in loan applications for consumer goods
        if is_ramadan:
            # During Ramadan, people tend to spend more, affecting debt ratios
            df['Payment_to_Income_Ratio'] = df['Payment_to_Income_Ratio'] * random.uniform(1.05, 1.15)
            
            # More applications for smaller loans during Ramadan
            small_loan_mask = df['Loan_Amount'] < 10000000  # Loans under 10M IDR
            df.loc[small_loan_mask, 'Default_Probability'] *= random.uniform(1.1, 1.2)

        # Apply year-end bonus effects
        if is_year_end:
            # Better repayment ability during bonus periods
            df['Payment_to_Income_Ratio'] = df['Payment_to_Income_Ratio'] * random.uniform(0.85, 0.95)

            # Lower default probability for all applicants during bonus periods
            df['Default_Probability'] *= random.uniform(0.8, 0.9)

            # Increase in loan applications for larger amounts
            large_loan_mask = df['Loan_Amount'] > 20000000 # Loans over 20M IDR
            df.loc[large_loan_mask, 'Default_Probability'] *= random.uniform(0.9, 1.0)

        # Cap probabilities
        df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Update loan status based on new probabilities
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )

        return df
    
    def apply_regulatory_changes(self, df, date):
        """
        Apply effects of regulatory changes from OJK (Financial Services Authority).

        Args:
            df: DataFrame containing loan application data
            date: Date string for when these regulations take effect
            
        Returns:
            DataFrame with regulatory change effects
        """
        df = df.copy()

        # Convert to datetime
        current_date = datetime.strptime(date, '%Y-%m-%d')

        # Simulate specific regulatory changes (example scenarios)

        # 1. Mid-2022: Stricter down payment requirements (affects approval probability)
        if current_date >= datetime.strptime('2022-06-01', '%Y-%m-%d'):
            # Higher loan amounts require higher income after regulation
            high_loan_mask = df['Loan_Amount'] > 15000000
            df.loc[high_loan_mask, 'Default_Probability'] *= 1.15

        # 2. Early 2023: Digital lending regulations (affects online vs offline)
        if current_date >= datetime.strptime('2023-01-15', '%Y-%m-%d'):
            if 'Application_Method' in df.columns:
                # Digital applications face more scrutiny
                digital_mask = df['Application_Method'] == 'Digital'
                df.loc[digital_mask, 'Default_Probability'] *= 1.1

        # 3. Late 2023: Improved rural financial inclusion initiatives
        if current_date >= datetime.strptime('2023-09-01', '%Y-%m-%d'):
            rural_masks = [
                df['Location'] == 'Java/Rural',
                df['Location'] == 'Kalimantan',
                df['Location'] == 'Sulawesi',
                df['Location'] == 'Other Islands'
            ]
            rural_mask = pd.Series(False, index=df.index)
            for mask in rural_masks:
                rural_mask = rural_mask | mask

            df.loc[rural_mask, 'Default_Probability'] *= 0.9

        # Cap probabilities
        df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Update loan status
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )

        return df
    
    def apply_market_evolution(self, df, date):
        """
        Apply effects of market evolution like digital lending growth
        and changing consumer preferences.

        Args:
            df: DataFrame containing loan application data
            date: Date string for the market conditions

        Returns:
            DataFrame with market evolution effects
        """

        df = df.copy()

        # Convert to datetime
        current_date = datetime.strptime(date, '%Y-%m-%d')
        reference_date = datetime.strptime('2021-01-01', '%Y-%m-%d')

        # Calculate time difference in months
        months_diff = (current_date - reference_date).days / 30.44

        # Digital adoption increases over time (if Application_Method exists)
        if 'Application_Method' in df.columns:
            # Calculate probability adjustment - digital adoption grows over time
            digital_growth_factor = min(0.3, 0.05 + (months_diff * 0.01))

            # Get random subset of physical applications to convert to digital
            physical_mask = df['Application_Method'] == 'Physical'
            convert_count = int(sum(physical_mask) * digital_growth_factor)

            if convert_count > 0:
                # Prioritize younger, more educated applicants for digital conversion
                convert_indices = df[physical_mask].sort_values(
                    by=['Age', 'Education_Level'],
                    ascending=[True, False]
                ).index[:convert_count]

                df.loc[convert_indices, 'Application_Method'] = 'Digital'

        # Older applicants start adopting digital more
        if months_diff > 18 and 'Application_Method' in df.columns:
            older_applicants = df[(df['Age'] > 45) & (df['Application_Method'] == 'Physical')]
            convert_count = int(len(older_applicants) * 0.15)

            if convert_count > 0:
                convert_indices = older_applicants.sample(convert_count, random_state=int(months_diff)).index
                df.loc[convert_indices, 'Application_Method'] = 'Digital'

        # Shift in consumer preferences toward longer loan terms
        if months_diff > 12:
            # Gradual shift toward 36-month loans from 24-month loans
            shift_prob = min(0.4, 0.1 + (months_diff - 12) * 0.01)
            twenty_four_month_mask = df['Loan_Term'] == 24

            shift_indices = df[twenty_four_month_mask].sample(
                frac=shift_prob,
                random_state=int(months_diff)
            ).index

            df.loc[shift_indices, 'Loan_Term'] = 36

            # Recalculate monthly payment
            df.loc[shift_indices, 'Monthly_Payment'] = df.loc[shift_indices, 'Loan_Amount'] / 36 * 1.1
            df.loc[shift_indices, 'Payment_to_Income_Ratio'] = (
                df.loc[shift_indices, 'Monthly_Payment'] / df.loc[shift_indices, 'Monthly_Income']
            )

        # Recalculate default probabilities based on new values
        # This is a simplified approach - in a real model, you'd rerun the probability calculation
        if months_diff > 6:
            df['Default_Probability'] = df['Default_Probability'] * random.uniform(0.95, 1.05)

        # Cap probabilities
        df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Update loan status
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )

        return df
    
    def generate_time_series_data(self, base_df, start_date, end_date, interval_days=30,
                                  drift_types=None, samples_per_period=1000):
        """
        Generate a time series of datasets with progressive drift.
        
        Args:
            base_df: Base DataFrame to start with
            start_date: Starting date for the time series
            end_date: Ending date for the time series
            interval_days: Days between generated datasets
            drift_types: List of drift types to apply, e.g., ['economic', 'seasonal']
                       If None, all drift types will be applied
            samples_per_period: Number of samples to generate for each time period
            
        Returns:
            List of (date, DataFrame) tuples representing the time series
        """
        # Convert date strings to datetime objects
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Default to all drift types if none specified
        all_drift_types = ['economic', 'seasonal', 'regulatory', 'market']
        drift_types = drift_types or all_drift_types

        # Generate time periods
        current = start
        time_series_data = []

        while current <= end:
            current_date = current.strftime('%Y-%m-%d')

            # Select samples for this period
            period_samples = base_df.sample(
                n=min(samples_per_period, len(base_df)),
                random_state=int(current.timestamp()) % 10000
            ).copy()

            # Set the application date to current period
            period_samples['Application_Date'] = current_date

            # Apply selected drift effects
            drifted_df = period_samples.copy()

            if 'economic' in drift_types:
                drifted_df = self.apply_economic_drift(drifted_df, current_date)
                
            if 'seasonal' in drift_types:
                drifted_df = self.apply_seasonal_effects(drifted_df, current_date)
                
            if 'regulatory' in drift_types:
                drifted_df = self.apply_regulatory_changes(drifted_df, current_date)
                
            if 'market' in drift_types:
                drifted_df = self.apply_market_evolution(drifted_df, current_date)

            # Add to time series
            time_series_data.append((current_date, drifted_df))

            # Move to next period
            current += timedelta(days=interval_days)

        return time_series_data
    
    def save_time_series_data(self, time_series_data, output_dir='data/synthetic/time_series'):
        """Save the time series data to files."""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Combine all periods into one file with time column
        all_data = []
        for date, df in time_series_data:
            period_df = df.copy()
            period_df['Period'] = date
            all_data.append(period_df)

        # Concatenate all periods
        combined_df = pd.concat(all_data, ignore_index=True)

        # Save combined file
        combined_file = os.path.join(output_dir, "loan_data_with_drift.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"Combined time series data saved to {combined_file}")

        # Also save individual period files
        for date, df in time_series_data:
            filename = f"loan_data_{date}.csv"
            file_path = os.path.join(output_dir, filename)
            df.to_csv(file_path, index=False)

        print(f"Individual period files saved to {output_dir}")
        return combined_file
    
if __name__ == "__main__":
    from data_generator import IndonesianLoanDataGenerator
    from bias_injector import BiasInjector

    # Generate base data
    print("Generating base data...")
    generator = IndonesianLoanDataGenerator()
    base_data = generator.generate_dataset(n_samples=20000)

    # Inject some initial bias
    print("Injecting initial bias...")
    injector = BiasInjector()
    biased_data = injector.inject_multiple_biases(
        base_data,
        biases=['geographic', 'gender', 'occupation'],
        severity=1.0
    )

    # Create drift simulator
    print("Setting up drift simulator...")
    drift_simulator = DataDriftSimulator()

    # Generate time series data with drift
    print("Generating time series data with drift...")
    time_series = drift_simulator.generate_time_series_data(
        biased_data,
        start_date='2021-01-01',
        end_date='2023-12-31',
        interval_days=90,  # Quarterly data
        samples_per_period=5000
    )

    # Save time series data
    print("Saving time series data...")
    drift_simulator.save_time_series_data(time_series)

    # Print some statistics
    print("\nTime series periods generated:")
    for date, df in time_series:
        default_rate = (df['Loan_Status'] == 'Default').mean()
        print(f"Period {date}: {len(df)} samples, Default rate: {default_rate:.2%}")
