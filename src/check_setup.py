import numpy as np
import pandas as pd
import sklearn 
import fastapi

print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"FastAPI version: {fastapi.__version__}")

df = pd.read_csv("data/animal_disease_dataset.csv")
print(f"DataFrame shape: {df.shape}")