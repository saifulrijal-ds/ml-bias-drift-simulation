# ML Bias & Drift Simulation

A comprehensive simulation environment to explore bias and data drift in the context of Indonesian motorcycle loan applications. This project:

1. Generates synthetic data representative of Indonesian loan applicants
2. Deliberately introduces various forms of bias
3. Simulates realistic data drift scenarios
4. Develops ML pipelines with fairness evaluation
5. Implements drift detection and monitoring

## Project Structure

- `data/` - Contains raw, processed, and synthetic datasets
- `src/` - Source code
  - `data_generation/` - Code for synthetic data generation
  - `preprocessing/` - Data cleaning and feature engineering
  - `models/` - ML model implementation
  - `evaluation/` - Fairness and performance evaluation
  - `visualization/` - Visualization scripts
- `notebooks/` - Jupyter notebooks for exploration and demonstration
- `config/` - Configuration files
- `tests/` - Unit and integration tests
- `docs/` - Documentation

## Setup and Installation

```bash
# Clone the repository
git clone [repository-url]
cd bfi-ml-bias-drift-simulation

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt