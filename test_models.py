from create_dataset import create_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from nn import train_nn
pd.set_option('future.no_silent_downcasting', True)

if __name__ == "__main__":

    # Create our data_frame from the data.csv file
    dataset, testset, data_frame = create_dataset()
    
    # Remove the columns that aren't relevant and are combined to get latent risk
    data_frame = data_frame.drop(columns=['MCC Code', 'Is Registered', 'Accepts Card','Average Rate','Average Fee'])

    # Print the correlation between all variables and Normalised Fees
    print(dataset.corr()['Normalised Fees'].sort_values(ascending=False))

    # Prepare dataset for training and validation
    X_data = dataset[['Latent Risk', 'Annual Card Turnover', 'Average Transaction Amount']]
    y_data = dataset['Normalised Fees']

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.1, random_state = 13)

    # Select the models we would like to test 
    models = {
        "Linear Regression": LinearRegression(),
        "Bayesian Ridge": BayesianRidge(),
        "Lasso": Lasso(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(loss='squared_error'),
        "Decision Tree Regressor": RandomForestRegressor(criterion='squared_error'),
        "Nearest Neighbor Regressor": KNeighborsRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor()
    }

    # Iterate through models and evaluate based on metrics
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        val_errors = y_val - y_pred
        
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print("Validation Mean Squared Error (MSE):", mse)
        print("Validation Mean Absolute Error (MAE):", mae)
        print("Validation R-squared (R2) Score:", r2)

    # Evaluate on simple neural net in same manner
    y_pred, y_val = train_nn(X_train,X_val,y_train,y_val)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"\nEvaluating NN..")
    print("Validation Mean Squared Error (MSE):", mse)
    print("Validation Mean Absolute Error (MAE):", mae)
    print("Validation R-squared (R2) Score:", r2)

