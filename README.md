## Steps:
1. Clone the GitHub

  git clone https://github.com/xingyao518/NYC311-Flooding-Bayesian-Model.git

2. Enter the project root directory

  cd ~/NYC311-Flooding-Bayesian-Model

3. Create a virtual environment

  python3 -m venv venv

4. Activate the virtual environment

  source venv/bin/activate

5. Install dependent packages

  pip install pandas geopandas shapely numpy scipy scikit-learn seaborn matplotlib cmdstanpy

6. Run the model

  python3 main_pipeline.py
