import csv
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# Example usage: unique_values = get_unique_items_from_column("data.csv", "MCC Code") - returns all unique MCC Codes in the data
def get_unique_items_from_column(filename, column_name):
    unique_values = set()

    with open(filename, mode='r', newline='', encoding='utf-8') as csv_file:
        data = csv.DictReader(csv_file)
        
        for row in data:
            value = row[column_name].strip()
            if value:
                try:
                    value = int(value)
                except ValueError:
                    if value == '.':
                        value = 'null' 
                unique_values.add(value)

    return sorted(unique_values)

# Based on evaluated risk of business categorisation this function will return a latent risk variable
def get_MCC_latentRisk(mcc_code):
    mcc_code = int(mcc_code)
    if 1 <= mcc_code <= 1499:
        return 1  # Low
    elif 1500 <= mcc_code <= 2999:
        return 2  # Medium
    elif 3000 <= mcc_code <= 3299:
        return 3  # High
    elif 3300 <= mcc_code <= 3499:
        return 3  # High
    elif 3500 <= mcc_code <= 3599:
        return 1  # Low
    elif 3600 <= mcc_code <= 3999:
        return 2  # Medium
    elif 4000 <= mcc_code <= 4799:
        return 3  # High
    elif 4800 <= mcc_code <= 4999:
        return 1  # Low
    elif 5000 <= mcc_code <= 5200:
        return 1  # Low
    elif 5200 <= mcc_code <= 5299:
        return 1  # Low
    elif 5300 <= mcc_code <= 5399:
        return 1  # Low
    elif 5400 <= mcc_code <= 5499:
        return 1  # Low
    elif 5500 <= mcc_code <= 5599:
        return 2 # Medium
    elif 5600 <= mcc_code <= 5699:
        return 1  # Low
    elif 5700 <= mcc_code <= 5799:
        return 1  # Low
    elif 5810 <= mcc_code <= 5819:
        return 1  # Low
    elif 5820 <= mcc_code <= 5829:
        return 1  # Low
    elif 5830 <= mcc_code <= 7299:
        return 2  # Medium
    elif 7300 <= mcc_code <= 7799:
        return 1  # Low
    elif 7800 <= mcc_code <= 7899:
        return 2  # Medium
    elif 7900 <= mcc_code <= 7999:
        return 3  # High
    elif 8000 <= mcc_code <= 8999:
        return 1  # Low
    elif 9000 <= mcc_code <= 9999:
        return 3  # High
    else:
        print(mcc_code)
        return 0 # Exception

# Create a unique id for each unique provider in the dataset and return it
def get_provider_code(provider):
    provider = str(provider)
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
    
    # edge cases
    if provider == "Barclaysâ€™s" or provider == "barclaycard":
        provider = "barclay"
    elif provider == "Izettle" or provider == "izettle":
        provider = "zettle"
    elif provider == "Paymentssense":
        provider = "Paymentsense"
    elif provider == "teya payment solutions":
        provider = "teya"
    elif provider == "accepts cards" or provider == "merchantservices" or provider == "Paymen":
        provider = "other"
    elif provider == "firstdata":
        provider = "null"

    tokens = provider.lower().split()
    for i, company in enumerate(companies):
        comp = company.lower().replace(" ","")
        if any(t in comp for t in tokens):
            # print(provider, companies[i])
            return i
    return -1

#  Function to calculate the average card fee a merchant will pay per transaction and the total annual fees the merchant will pay on all transactions based on the brief's assumptions
def get_fees(row):

    annual_turnover = row["Annual Card Turnover"]
    avg_transaction = row["Average Transaction Amount"]

    fees = []
    row = row.iloc[-6:]
    for fee in row: 
        fee = fee.strip()
        percent_index = fee.index('%')
        rate_str = fee[:percent_index].strip()
        rate = float(rate_str) / 100
    
        plus_index = fee.index('+')
        p_index = fee.index('p')
        fee_str = fee[plus_index + 2 : p_index].strip()
        flat_fee = float(fee_str) / 100.0
        fees.append([rate,flat_fee])

    average_rate = round(0.4*0.9*fees[0][0] + 0.6*0.9*fees[1][0]  + 0.4*0.08*fees[2][0] + 0.6*0.08*fees[3][0] + 0.4*0.02*fees[4][0] + 0.6*0.02*fees[5][0],6)
    average_fee = round(0.4*0.9*fees[0][1] + 0.6*0.9*fees[1][1]  + 0.4*0.08*fees[2][1] + 0.6*0.08*fees[3][1] + 0.4*0.02*fees[4][1] + 0.6*0.02*fees[5][1],6)
    calculated_fees = (average_rate*annual_turnover + average_fee*(annual_turnover/avg_transaction))/annual_turnover

    return average_rate, average_fee, calculated_fees

# Function to generate a latent risk variable based on merchant data
def get_latent_risk(row):

    mcc_risk = row['MCC Code']
        
    if row['Annual Card Turnover'] <= 9999:
        turnover_risk = 3
    elif 10000 <= row['Annual Card Turnover'] <= 99999:
        turnover_risk = 2
    else:  
        turnover_risk = 1

    if row['Average Transaction Amount'] <= 49.99:
        transaction_risk = 3
    elif 50 <= row['Average Transaction Amount'] <= 100:
        transaction_risk = 2
    else:
        transaction_risk = 1

    is_registed = 3*row["Is Registered"]
    if is_registed == 0:
        is_registed = 1
    accepts_card = 3*(1-row["Accepts Card"])
    if accepts_card == 0:
        accepts_card = 1

    risk = (is_registed+accepts_card+mcc_risk+transaction_risk+turnover_risk)/5
    
    return risk

# Function to return requried dataframes from data.csv
def create_dataset():
    # DATA PREPARATION
    data_frame = pd.read_csv("data.csv")

    columns_to_convert = ["Is Registered", "Accepts Card"]
    data_frame[columns_to_convert] = data_frame[columns_to_convert].replace({"Yes": 1, "No": 0})
    data_frame["MCC Code"] = data_frame["MCC Code"].apply(get_MCC_latentRisk)
    data_frame["Current Provider"] = data_frame["Current Provider"].apply(get_provider_code)
    data_frame[["Average Rate","Average Fee", "Normalised Fees"]] = data_frame.apply(get_fees, axis=1, result_type='expand')

    data_frame = data_frame.drop(columns=data_frame.columns[-9:][:-3])

    data_frame["Latent Risk"] = data_frame.apply(get_latent_risk, axis=1, result_type='expand')


    # sns.distplot(np.log(data_frame['Average Rate']))
    # plt.title("Frequency of Average Fee")
    # plt.show()

    # sns.barplot(x=data_frame["MCC Code"], y=data_frame['Average Rate'])
    # plt.title("Spread of Fee vs MCC Risk")
    # plt.xticks(rotation="vertical")
    # plt.show()

    # sns.scatterplot(x=data_frame['Latent Risk'],y=data_frame['Normalised Fees'])
    # plt.title("Log of Annual Card Turnover vs Average Rate")
    # plt.show()

    # print(data_frame.corr()['Average Rate'].sort_values(ascending=False))

    dataset = data_frame[(data_frame["Current Provider"] != -1) & (data_frame["Accepts Card"] == 1)]
    testset = data_frame[(data_frame["Current Provider"] == -1) | (data_frame["Accepts Card"] != 1)]

    return dataset, testset, data_frame

# Function to return a datapoint (single point dataframe) from floom.csv
def create_datapoint(csv_path):
    data_frame = pd.read_csv(csv_path)

    columns_to_convert = ["Is Registered", "Accepts Card"]
    data_frame[columns_to_convert] = data_frame[columns_to_convert].replace({"Yes": 1, "No": 0})
    data_frame["MCC Code"] = data_frame["MCC Code"].apply(get_MCC_latentRisk)
    data_frame["Current Provider"] = data_frame["Current Provider"].apply(get_provider_code)
    data_frame[["Average Rate","Average Fee", "Normalised Fees"]] = data_frame.apply(get_fees, axis=1, result_type='expand')

    data_frame = data_frame.drop(columns=data_frame.columns[-9:][:-3])

    data_frame["Latent Risk"] = data_frame.apply(get_latent_risk, axis=1, result_type='expand')

    return data_frame

if __name__ == "__main__":
    dataset, testset, data_frame = create_dataset()

    print(data_frame.corr()['Normalised Fees'].sort_values(ascending=False))







