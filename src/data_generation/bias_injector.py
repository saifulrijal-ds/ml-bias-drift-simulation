"""
Bias Injector for Indonesian Loan Application Data

This module contains functions to deliberately introduce various types of
bias into the synthetic loan application data, simulating real-world biases
that might exist in the Indonesian financial context.
"""

import pandas as pd
import numpy as np
import random

class BiasInjector:
    """
    Introduces various forms of bias into loan application datasets,
    simulating real-world biases in the Indonesian finance context.
    """

    def __init__(self, random_seed=42):
        """Initialize the bias injector with a random seed for reproducibility."""
        np.random.seed(random_seed)
        random.seed(random_seed)

    def inject_geographic_bias(self, df, severity=1.0):
        """
        Introduce geographic bias where applicants from certain regions
        have different probabilities of loan approval.
        
        Args:
            df: DataFrame containing loan application data
            severity: Controls the strength of the bias (0.0-2.0)
                      1.0 is the default level, <1.0 reduces bias, >1.0 increases it
                      
        Returns:
            DataFrame with injected geographic bias
        """

        df = df.copy()

        # Define location default multipliers (higher = more likely to default)
        location_default_multipliers = {
            'Java/Urban': 0.7,       # Urban Java gets preferential treatment
            'Java/Rural': 1.0,       # Baseline reference
            'Sumatra': 1.2 * severity,
            'Kalimantan': 1.4 * severity,
            'Sulawesi': 1.5 * severity,
            'Other Islands': 1.6 * severity  # Remote islands most disadvantaged
        }

        # Apply multipliers to adjust default probability
        for location, multiplier in location_default_multipliers.items():
            mask = df['Location'] == location
            df.loc[mask, 'Default_Probability'] = df.loc[mask, 'Default_Probability'] * multiplier

            # Cap probabilities between 0.01 and 0.95
            df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Regenerate loan status based on new probabilities
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )
        
        return df
    
    def inject_gender_bias(self, df, severity=1.0):
        """
        Introduce gender bias in loan approval process.
        
        Args:
            df: DataFrame containing loan application data
            severity: Controls the strength of the bias (0.0-2.0)
            
        Returns:
            DataFrame with injected gender bias
        """
        df = df.copy()

        # Different income thresholds based on gender
        # For the same default probability, women need to show higher income
        income_adjustment = {
            'Male': 1.0,                        # Baseline
            'Female': 1.0 + (0.3 * severity)    # Women need higher income for same approval odds
        }

        # Apply income adjustment to default probability calculation
        for gender, factor in income_adjustment.items():
            mask = df['Gender'] == gender

            # Adjust default probability based on gender
            # For women, make the same income level appear less sufficient (higher default risk)
            if gender == 'Female':
                df.loc[mask, 'Default_Probability'] = df.loc[mask, 'Default_Probability'] * factor

        # Cap probabilities
        df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Regenerate loan status
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )

        return df
    
    def inject_digital_access_bias(self, df, severity=1.0):
        """
        Introduce bias related to digital access/application method.
        
        Args:
            df: DataFrame containing loan application data
            severity: Controls the strength of the bias
            
        Returns:
            DataFrame with digital access bias
        
        """
        df = df.copy()

        # Create application method feature (digital vs. physical)
        # More likely to be digital in urban areas and among younger people
        df['Application_Method'] = 'Physical'  # Default

        # Urban locations more likely to use digital
        urban_digital_prob = {
            'Java/Urban': 0.85,
            'Java/Rural': 0.40,
            'Sumatra': 0.60,
            'Kalimantan': 0.45,
            'Sulawesi': 0.40,
            'Other Islands': 0.25
        }

        for location, prob in urban_digital_prob.items():
            mask = df['Location'] == location
            n_digital = int(np.sum(mask) * prob)

            # Get indices for this location, prioritizing younger and more educated
            location_indices = df[mask].sort_values(
                by=['Education_Level', 'Age'], 
                ascending=[False, True]
            ).index[:n_digital]
            
            df.loc[location_indices, 'Application_Method'] = 'Digital'
        
        # Apply bias: digital applications get preferential treatment
        if severity > 0:
            digital_advantage = 0.75 - (0.25 * severity)  # Lower probability of default for digital
            physical_penalty = 1.0 + (0.25 * severity)    # Higher probability for physical
            
            df.loc[df['Application_Method'] == 'Digital', 'Default_Probability'] *= digital_advantage
            df.loc[df['Application_Method'] == 'Physical', 'Default_Probability'] *= physical_penalty
            
            # Cap probabilities
            df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)
            
            # Regenerate loan status
            df['Loan_Status'] = df['Default_Probability'].apply(
                lambda x: 'Default' if random.random() < x else 'Fully Paid'
            )

        return df
    
    def inject_occupation_bias(self, df, severity=1.0):
        """
        Introduce bias that favors formal employees and disadvantages
        gig economy workers and agricultural workers.
        
        Args:
            df: DataFrame containing loan application data
            severity: Controls the strength of the bias
            
        Returns:
            DataFrame with occupation bias
        """

        df = df.copy()

        # Define occupation default multipliers
        occupation_default_multipliers = {
            'Formal Employee': 0.7,               # Advantage for formal employees
            'Entrepreneur': 1.0,                  # Baseline reference
            'Gig Worker': 1.3 * severity,         # Disadvantage for gig workers
            'Agricultural': 1.5 * severity        # Largest disadvantage for agricultural workers
        }

        # Apply multipliers to default probability
        for occupation, multiplier in occupation_default_multipliers.items():
            mask = df['Occupation_Type'] == occupation
            df.loc[mask, 'Default_Probability'] = df.loc[mask, 'Default_Probability'] * multiplier

        # Cap probabilities
        df['Default_Probability'] = df['Default_Probability'].clip(0.01, 0.95)

        # Regenerate loan status
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )

        return df
    
    def inject_multiple_biases(self, df, biases=None, severity=1.0):
        """
        Apply multiple biases at once.

        Args:
            df: DataFrame containing loan application data
            biases: List of bias types to apply, e.g., ['geographic', 'gender']
                   If None, all biases will be applied
            severity: Controls the strength of all biases
            
        Returns:
            DataFrame with multiple injected biases
        """
        df = df.copy()

        # Default to all bias types if none specified
        all_biases = ['geographic', 'gender', 'digital_access', 'occupation']
        biases = biases or all_biases

        # Apply each requested bias type
        biased_df = df.copy()

        if 'geographic' in biases:
            biased_df = self.inject_geographic_bias(biased_df, severity)
            
        if 'gender' in biases:
            biased_df = self.inject_gender_bias(biased_df, severity)
            
        if 'digital_access' in biases:
            biased_df = self.inject_digital_access_bias(biased_df, severity)
            
        if 'occupation' in biases:
            biased_df = self.inject_occupation_bias(biased_df, severity)
        
        return biased_df
    
    def save_biased_dataset(self, df, bias_types, severity, output_dir='data/synthetic'):
        """Save the biased dataset with descriptive filename."""
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        bias_label = '_'.join(bias_types) if isinstance(bias_types, list) else bias_types
        filename = f"biased_data_{bias_label}_severity{severity:.1f}.csv"
        file_path = os.path.join(output_dir, filename)

        df.to_csv(file_path, index=False)
        print(f"Biased dataset saved to {file_path}")
        return file_path
    
    
if __name__ == "__main__":
    from data_generator import IndonesianLoanDataGenerator
    
    # Generate baseline data
    generator = IndonesianLoanDataGenerator()
    baseline_data = generator.generate_dataset(n_samples=5000)
    
    # Create bias injector
    injector = BiasInjector()
    
    # Example: Generate datasets with different bias types
    geographic_biased = injector.inject_geographic_bias(baseline_data)
    gender_biased = injector.inject_gender_bias(baseline_data)
    digital_biased = injector.inject_digital_access_bias(baseline_data)
    occupation_biased = injector.inject_occupation_bias(baseline_data)
    
    # Example: Generate data with multiple biases
    multi_biased = injector.inject_multiple_biases(
        baseline_data, 
        biases=['geographic', 'gender'],
        severity=1.5
    )
    
    # Save datasets
    injector.save_biased_dataset(geographic_biased, "geographic", 1.0)
    injector.save_biased_dataset(gender_biased, "gender", 1.0)
    injector.save_biased_dataset(digital_biased, "digital", 1.0)
    injector.save_biased_dataset(occupation_biased, "occupation", 1.0)
    injector.save_biased_dataset(multi_biased, ["geographic", "gender"], 1.5)
    
    # Print some statistics to verify bias injection
    print("\nBaseline default rates:")
    print(baseline_data.groupby('Gender')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))
    
    print("\nGender-biased default rates:")
    print(gender_biased.groupby('Gender')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))
    
    print("\nGeographic-biased default rates:")
    print(geographic_biased.groupby('Location')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))

    print("\nDigital-biased default rates:")
    print(digital_biased.groupby('Gender')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))

    print("\Occupation-biased default rates:")
    print(occupation_biased.groupby('Gender')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))

    print("\Multiple-biased default rates:")
    print(multi_biased.groupby('Gender')['Loan_Status'].apply(
        lambda x: (x == 'Default').mean()
    ))
