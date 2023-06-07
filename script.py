from comet_ml import Experiment
import comet_ml
import torch
import torchvision
from comet_ml.integration.pytorch import log_model
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

experiment = Experiment(
  api_key = "Tcgb2u9Wu6OHaXqB8BNfs090x",
  project_name = "general",
  workspace="1272371"
)

AUTH = {
    'API_Key':"9fn6Ayv5jAAqBTwpIOQi2q41n",
    'API_Key_secret':"vOwdTEEF5fzg5XZqpaHBQiglti1PubugbH7t2HuwYxFzyeifDl",
    'Bearer_Token':"AAAAAAAAAAAAAAAAAAAAAPMZoAEAAAAAOfaAHqNi38FFzZBgNQwvjWnAuuI%3DSA40ROMKVqhMSWuhXGKJQWy6P9Z9OrcEEO0tkirmXXbMfCnnwG",
    'Access_Toekn':"1500956228180189193-ZS5eMTedS4bt0HOIKCAChZvTaXycZc",
    'Access_Token_secret':"bet4LnQ9A5DkxMNS2Gsz9xPJn1WBma8vSmAvvqwiVYmui"
}
# Report multiple hyperparameters using a dictionary:
hyper_params = {
   "learning_rate": 0.5,
   "steps": 100000,
   "batch_size": 50,
}
experiment.log_parameters(hyper_params)

# Generate regression data
X, y = make_regression(n_samples=100, n_features=1, noise=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Seamlessly log your PyTorch model
log_model(experiment, model, model_name="TheLinearRegressionModel")
