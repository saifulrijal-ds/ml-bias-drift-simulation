"""
Data Generator for Indonesian Motorcycle Loan Applications

This module creates synthetic data representing motorcycle loan applicants
in Indonesia, with specific attention to demographic and financial attributes
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
import os
from datetime import datetime, timedelta

# Initialize Faker with Indonesian locale
fake = Faker('id_ID')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

class IndonesianLoanDataGenerator:
    """
    Generates synthetic data for Indonesian motorcycle loan applications
    with realistic demographic and financial attributes.
    """

    def __init__(self):
        # Define demographic distributions based on Indonesian context
        self.locations = {
            'Java/Urban': 0.45,     # 45% from urban Java
            'Java/Rural': 0.25,     # 25% from rural Java
            'Sumatra': 0.15,        # 15% from Sumatra
            'Kalimantan': 0.06,     # 6% from Kalimantan
            'Sulawesi': 0.05,       # 5% from Sulawesi
            'Other Islands': 0.04   # 4% from other islands
        }

        self.education_levels = {
            'Elementary': 0.10,
            'Secondary': 0.25,
            'High School': 0.40,
            'Diploma': 0.15,
            'University': 0.10
        }

        self.occupation_types = {
            'Formal Employee': 0.40,
            'Entrepreneur': 0.30,
            'Gig Worker': 0.20,
            'Agricultural': 0.10
        }

        self.bank_account_types = {
            'None': 0.20,
            'Basic': 0.60,
            'Premium': 0.20
        }

        # Define income distributions by occupation and education
        self.income_params = {
            'Formal Employee': {'base': 5000000, 'std': 3000000, 'edu_factor': 1.3},
            'Entrepreneur': {'base': 4500000, 'std': 4000000, 'edu_factor': 1.2},
            'Gig Worker': {'base': 3000000, 'std': 1500000, 'edu_factor': 1.1},
            'Agricultural': {'base': 2500000, 'std': 1000000, 'edu_factor': 1.05}
        }

        # Define default probability factors
        self.default_base_rate = 0.08 # 8% base default rate

    def _generate_age(self):
        """Generate age with realistic distribution for loan applicants."""
        # Age distribution skewed toward young and middle-aged adults
        return int(np.random.triangular(18, 30, 70))
    
    def _generate_income(self, occupation, education):
        """Generate monthly income based on occupation and education."""
        params = self.income_params[occupation]
        base = params['base']
        std = params['std']

        # Education factor increases income
        edu_multiplier = 1.0
        if education == 'University':
            edu_multiplier = params['edu_factor'] * 1.5
        elif education == 'Diploma':
            edu_multiplier = params['edu_factor'] * 1.2
        elif education == 'High School':
            edu_multiplier = params['edu_factor'] * 1.0
        elif education == 'Secondary':
            edu_multiplier = params['edu_factor'] * 0.8
        else:  # Elementary
            edu_multiplier = params['edu_factor'] * 0.6

        # Generate income with some randomness
        income = np.random.normal(base * edu_multiplier, std)
        return max(1500000, int(income)) # Minimum wage-like floor
    
    def _generate_loan_amount(self, income):
        """Generate loan amount requested based on income."""
        # Motorcycle loans typically range from 5-25 million IDR
        # Higher income allows higher loan amounts
        income_factor = min(5, max(1, income / 3000000))
        mean_loan = 10000000 * income_factor
        std_loan = 5000000

        loan_amount = int(np.random.normal(mean_loan, std_loan))
        # Ensure loan is between 5M and 30M IDR
        return max(5000000, min(30000000, loan_amount))
    
    def _calculate_default_probability(self, row):
        """Calculate the probability of default based on applicant attributes."""
        prob = self.default_base_rate

        # Income relative to loan amount
        pti_ratio = row['Payment_to_Income_Ratio']
        if pti_ratio > 0.5:
            prob *= 2.5
        elif pti_ratio > 0.3:
            prob *= 1.5
        elif pti_ratio < 0.2:
            prob *= 0.7

        # Age factors
        if row['Age'] < 25:
            prob *= 1.3
        elif row['Age'] > 50:
            prob *= 1.1

        # Employment stability
        if row['Years_at_Employment'] < 1:
            prob *= 1.5
        elif row['Years_at_Employment'] > 5:
            prob *= 0.7

        # Address stability
        if row['Years_at_Address'] < 1:
            prob *= 1.2
        elif row['Years_at_Address'] > 5:
            prob *= 0.8

        # Existing debt
        if row['Existing_Debt_Amount'] > row['Monthly_Income'] * 3:
            prob *= 1.8

        # Prior loans history
        if row['Prior_Loans'] == 'Yes' and row['Payment_History'] == 'Good':
            prob *= 0.6
        elif row['Prior_Loans'] == 'Yes' and row['Payment_History'] == 'Poor':
            prob *= 2.0

        # Bank account
        if row['Bank_Account_Type'] == 'None':
            prob *= 1.3
        elif row['Bank_Account_Type'] == 'Premium':
            prob *= 0.8
        
        # Education level
        if row['Education_Level'] == 'Elementary':
            prob *= 1.2
        elif row['Education_Level'] == 'University':
            prob *= 0.8

        # Cap the probability
        return min(0.95, max(0.01, prob))
    
    def generate_dataset(self, n_samples=10000, start_date='2021-01-01', end_date='2022-12-31'):
        """
        Generate a synthetic dataset of Indonesian motorcycle loan applications.

        Args:
            n_samples: Number of loan applications to generate
            start_date: Starting date for application timestamps
            end_date: Ending date for application timestamps

        Returns:
            pandas.DataFrame: DataFrame containing synthetic loan application data
        """

        data = []

        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end_date - start_date).days

        for _ in range(n_samples):
            # Generate demographic information
            gender = random.choice(['Male', 'Female'])
            age = self._generate_age()

            location = random.choices(
                list(self.locations.keys()),
                list(self.locations.values())
            )[0]

            education = random.choices(
                list(self.education_levels.keys()),
                list(self.education_levels.values())
            )[0]

            occupation = random.choices(
                list(self.occupation_types.keys()),
                list(self.occupation_types.values())
            )[0]

            # Generate time-related features
            years_at_address = min(age - 18, random.randint(0, 20))
            years_at_employment = min(age - 18, random.randint(0, 15))

            # Generate financial information
            monthly_income = self._generate_income(occupation, education)
            loan_amount = self._generate_loan_amount(monthly_income)

            loan_term = random.choice([12, 24, 36, 48])
            monthly_payment = loan_amount / loan_term * 1.1  # Simple interest approximation

            payment_to_income_ratio = monthly_payment / monthly_income
            existing_debt_amount = max(0, int(np.random.normal(monthly_income * 1.5, monthly_income * 0.8)))

            # Generate credit history
            prior_loans = random.choice(['Yes', 'No'])
            payment_history = 'None'
            if prior_loans == 'Yes':
                payment_history = random.choices(['Good', 'Poor'], [0.8, 0.2])[0]

            bank_account = random.choices(
                list(self.bank_account_types.keys()),
                list(self.bank_account_types.values())
            )[0]

            credit_card = random.choice(['Yes', 'No']) if bank_account != 'None' else 'No'

            # Generate application date
            random_days = random.randint(0, date_range)
            application_date = start_date + timedelta(days=random_days)

            # Marital status and dependents
            marital_status = random.choice(['Single', 'Married', 'Divorced', 'Widowed'])
            num_dependents = 0
            if marital_status in ['Married', 'Divorced', 'Widowed']:
                num_dependents = random.randint(0, 5)
            
            # Create the applicant record
            applicant = {
                'Application_Date': application_date.strftime('%Y-%m-%d'),
                'Age': age,
                'Gender': gender,
                'Location': location,
                'Education_Level': education,
                'Occupation_Type': occupation,
                'Marital_Status': marital_status,
                'Number_of_Dependents': num_dependents,
                'Years_at_Address': years_at_address,
                'Years_at_Employment': years_at_employment,
                'Monthly_Income': monthly_income,
                'Loan_Amount': loan_amount,
                'Loan_Term': loan_term,
                'Monthly_Payment': monthly_payment,
                'Payment_to_Income_Ratio': payment_to_income_ratio,
                'Existing_Debt_Amount': existing_debt_amount,
                'Prior_Loans': prior_loans,
                'Payment_History': payment_history,
                'Bank_Account_Type': bank_account,
                'Credit_Card': credit_card,
            }
            
            data.append(applicant)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Calculate default probability
        df['Default_Probability'] = df.apply(self._calculate_default_probability, axis=1)

        # Generate loan outcome based on default probability
        df['Loan_Status'] = df['Default_Probability'].apply(
            lambda x: 'Default' if random.random() < x else 'Fully Paid'
        )
        
        return df
    
    def save_dataset(self, df, filename, output_dir='data/synthetic'):
        """Save the generated dataset to a CSV file."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")
        return file_path

if __name__ == "__main__":
    # Example usage
    generator = IndonesianLoanDataGenerator()
    
    # Generate baseline dataset
    print("Generating baseline dataset...")
    baseline_data = generator.generate_dataset(n_samples=10000)
    generator.save_dataset(baseline_data, "baseline_loan_data.csv")
    
    # Print dataset statistics
    print("\nDataset Summary:")
    print(f"Total samples: {len(baseline_data)}")
    print(f"Default rate: {(baseline_data['Loan_Status'] == 'Default').mean():.2%}")
    
    # Print distribution of key fields
    print("\nLocation Distribution:")
    print(baseline_data['Location'].value_counts(normalize=True))
    
    print("\nOccupation Distribution:")
    print(baseline_data['Occupation_Type'].value_counts(normalize=True))
    
    print("\nGender Distribution:")
    print(baseline_data['Gender'].value_counts(normalize=True))
