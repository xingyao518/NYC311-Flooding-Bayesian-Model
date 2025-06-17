# Clone the GitHub
git clone https://github.com/xingyao518/NYC311-Flooding-Bayesian-Model.git

# Enter the project root directory
cd ~/NYC311-Flooding-Bayesian-Model

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependent packages
pip install pandas geopandas shapely numpy scipy scikit-learn matplotlib cmdstanpy

# Run the model
python3 main_pipeline.py
