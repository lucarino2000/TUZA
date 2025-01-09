from create_dataset import create_dataset, create_datapoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)

def label_price(row, std):
    if row["Test Error"] > std:
        return 'Un-Competitive'
    elif abs(row["Test Error"]) <= std:
        return 'Neutral'
    else:
        return 'Competitive'
    
def get_provider_name(code):
    companies = [
        "null",
        "adyen",
        "allied irish",
        "bambora",
        "barclay",
        "cardsaver",
        "castles",
        "clover",
        "dna",
        "dojo",
        "epos",
        "elavon",
        "everypay",
        "evo",
        "fresha",
        "globalpayments",
        "handepay",
        "hsbc",
        "kroo",
        "lloyds",
        "lopay",
        "nationwide",
        "natwest",
        "other",
        "payatrader",
        "payeat",
        "payment safe",
        "payment save",
        "payment sense",
        "paypal",
        "paypoint",
        "revolut",
        "shopify",
        "spire",
        "square",
        "stripe",
        "sumup",
        "take payments",
        "tap and go",
        "teya",
        "tide",
        "virgin",
        "wise",
        "world pay",
        "xln",
        "zettle"
    ]
    
    # Validate code
    if 0 <= code < len(companies):
        return companies[code]
    return "null"

if __name__ == "__main__":

    # Create our data_frame from the data.csv file
    dataset, testset, data_frame = create_dataset()
    
    # Remove the columns that aren't relevant or are combined to get latent risk
    data_frame = data_frame.drop(columns=['MCC Code', 'Is Registered', 'Accepts Card','Average Rate','Average Fee'])

    # Preparing datasets
    X_data = dataset[['Latent Risk', 'Annual Card Turnover', 'Average Transaction Amount']]
    y_data = dataset['Normalised Fees']
    X_test = testset[['Latent Risk', 'Annual Card Turnover', 'Average Transaction Amount']]
    y_test = testset['Normalised Fees']

    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size = 0.1, random_state = 13)

    # Fitting our chosen model
    gbr = GradientBoostingRegressor(loss='squared_error')
    gbr.fit(X_train, y_train)

    # Label the training data based on training error (train and validation set combined)
    y_train_pred = gbr.predict(X_data)
    std_labels = np.sqrt(pd.Series.var(abs(y_data-y_train_pred)))
    train_errors = y_data-y_train_pred

    dataset["Predicted Normalised Fees"] = y_train_pred
    dataset["Test Error"] = train_errors
    dataset['Price Label'] = dataset.apply(label_price, axis=1, std=std_labels)

    # Validation set
    y_pred = gbr.predict(X_val)
    val_errors = y_val - y_pred
    
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("Validation Mean Squared Error (MSE):", mse)
    print("Validation Mean Absolute Error (MAE):", mae)
    print("Validation R-squared (R2) Score:", r2)

    # Test set
    y_test_pred = gbr.predict(X_test)

    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("\nTesting Mean Squared Error (MSE):", mse_test)
    print("Testing Mean Absolute Error (MAE):", mae_test)
    print("Testing R-squared (R2) Score:", r2_test)

    test_errors = y_test - y_test_pred
    testset["Predicted Normalised Fees"] = y_test_pred
    testset["Test Error"] = test_errors
    
    # Creating a normal distribution plot to view the test errors vs. the validation std
    std = np.sqrt(pd.Series.var(abs(y_val-y_pred)))
    mae = np.mean(abs(test_errors))

    x = np.linspace(mae - 5*std, mae + 5*std, 1000)
    pdf = norm.pdf(x, loc=0, scale=std)

    bins = np.linspace(min(test_errors), max(test_errors), 100)
    hist, bin_edges = np.histogram(test_errors, bins=bins)

    pdf_scaled = pdf * max(hist) / max(pdf)

    plt.plot(x, pdf_scaled, label="Normal Distribution of Errors", color="blue")
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='red', alpha=0.7, label="Error Frequency")

    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error")
    plt.ylabel("Density")
    plt.axvline(x=std, color='green', linestyle='--', label='+1 Std Dev')
    plt.axvline(x=-std, color='orange', linestyle='--', label='-1 Std Dev')
    plt.legend()
    plt.show()

    # Apply labels to the data set and include necesaary information 
    testset['Price Label'] = testset.apply(label_price, axis=1, std=std)
    testset = testset[['Annual Card Turnover','Average Transaction Amount','Current Provider','Latent Risk','Normalised Fees', 'Predicted Normalised Fees', 'Price Label']]
    testset['Original Row Index'] = testset.index
    testset = testset[['Original Row Index','Annual Card Turnover','Average Transaction Amount','Current Provider','Latent Risk','Normalised Fees', 'Predicted Normalised Fees', 'Price Label']]

    # Combining the data and test sets to recreate the original dataframe now with the labeled data for observational and plotting purposes
    dataset = dataset[['Annual Card Turnover','Average Transaction Amount','Current Provider','Latent Risk','Normalised Fees', 'Predicted Normalised Fees', 'Price Label']]
    dataset['Original Row Index'] = dataset.index
    dataset = dataset[['Original Row Index','Annual Card Turnover','Average Transaction Amount','Current Provider','Latent Risk','Normalised Fees', 'Predicted Normalised Fees', 'Price Label']]

    data_frame = pd.concat([dataset, testset])
    data_frame = data_frame.sort_values(by='Original Row Index')
    data_frame = data_frame.reset_index(drop=True)
    data_frame["Current Provider"] = data_frame["Current Provider"].apply(get_provider_name)
    data_frame["Predicted Normalised Fees"] = data_frame["Predicted Normalised Fees"].apply(lambda x: round(x, 6))
    data_frame["Normalised Fees"] = data_frame["Normalised Fees"].apply(lambda x: round(x, 6))

    testset.to_csv("testset_discriminative_predictions.csv", index=False)
    data_frame.to_csv("data_labelled.csv", index=False)

    # Plot the classifcatios for each unique providers
    plot_data_frame = data_frame[data_frame["Current Provider"] != "null"]
    label_counts = plot_data_frame.groupby("Current Provider")["Price Label"].value_counts().unstack(fill_value=0)
    label_counts.plot(kind='bar', figsize=(10, 6))
    plt.title("Label Count for Each Unique Current Provider")
    plt.xlabel("Current Provider")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Price Label")
    plt.tight_layout()
    plt.show()

    # Example of Inference 
    datapoint = create_datapoint("floom.csv")
    predicted_price = gbr.predict(datapoint[['Latent Risk', 'Annual Card Turnover', 'Average Transaction Amount']])
    difference = datapoint['Normalised Fees'].values - predicted_price

    if difference > std:
        print("\nFloom Creative's card price is: Un-Competitive")
    elif abs(difference) <= std:
         print("\nFloom Creative's card price is: Neutral")
    else:
        print("\nFloom Creative's card price is: Competitive")


