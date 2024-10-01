import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    uploaded_file_performance = st.file_uploader("Upload your Performance Data CSV", key="performance")
    uploaded_file_fundamentals = st.file_uploader("Upload your Fundamentals Data CSV", key="fundamentals")
    uploaded_file_price = st.file_uploader("Upload your Price Data CSV", key="price")
    if uploaded_file_performance and uploaded_file_fundamentals and uploaded_file_price:
        performance_data = pd.read_csv(uploaded_file_performance)
        fundamentals_data = pd.read_csv(uploaded_file_fundamentals)
        price_data = pd.read_csv(uploaded_file_price)
        return performance_data, fundamentals_data, price_data
    return None, None, None

def process_data(performance_data, fundamentals_data, price_data):
    # Merge data
    merged_data = performance_data.merge(fundamentals_data, on='Name').merge(price_data[['Name', 'Last']], on='Name')

    # Data cleaning and conversions
    merged_data['Last'] = merged_data['Last'].replace({',': ''}, regex=True).astype(float)
    # merged_data['Market Cap'] = merged_data['Market Cap'].replace({'B': '', 'M': ''}, regex=True).astype(float)
    # merged_data['Market Cap'] = merged_data['Market Cap'].apply(lambda x: x / 1000 if x > 100 else x)
    merged_data['Revenue Score'] = merged_data['Revenue'].replace({'B': '', 'M': ''}, regex=True).astype(float)
    merged_data['Revenue Score'] = merged_data['Revenue Score'].apply(lambda x: x / 1000 if x < 100 else x)
    merged_data['P/E Score'] = 1 / merged_data['P/E Ratio']

    return merged_data


# Assuming data has already been loaded and cleaned as in previous steps
def buffet_strategy(merged_data):
    # Inverting P/E Ratio for scoring (lower is better, so invert to fit ascending score paradigm)
    # Only consider positive P/E ratios
    merged_data['P/E Ratio'] = merged_data['P/E Ratio'].apply(lambda x: x if x > 0 else 0)
    merged_data['P/E Score'] = merged_data['P/E Ratio'].apply(lambda x: 1 / x if x > 0 else 0)

    # Normalizing values
    scaler = MinMaxScaler()
    columns_to_normalize = ['3 Years', 'Revenue Score', 'P/E Score']
    merged_data[columns_to_normalize] = scaler.fit_transform(merged_data[columns_to_normalize])

    # Adjust weights according to Buffett's philosophy
    weights = {
        '3 Years': 0.30,   # Strong long-term performance
        'Revenue Score': 0.40,   # High revenue indicating strong business
        'P/E Score': 0.30  # Low P/E ratio indicating undervaluation
    }

    # Calculate scores
    merged_data['Buffett Score'] = merged_data[columns_to_normalize].dot(pd.Series(weights))

    # Return the sorted list of companies
    return merged_data.sort_values(by='Buffett Score', ascending=False).reset_index(drop=True)


# import requests

# def get_news_about(company_name):

#     url = ('https://newsapi.org/v2/everything?'
#         f'q={company_name}&'
#         'from=2024-09-29&'
#         'sortBy=popularity&'
#         'apiKey=6284d539d651404fad9168a3d36ec004')

#     response = requests.get(url)
#     return response.json().get('articles')

st.title('Moroccan Stock Analysis Tool')
performance_data, fundamentals_data, price_data = load_data()

if st.button("Analyze Stocks"):
    if performance_data is not None and fundamentals_data is not None and price_data is not None:
        merged_data = process_data(performance_data, fundamentals_data, price_data)
        scored_data = buffet_strategy(merged_data)
        # scored_data['News'] = scored_data['Name'].apply(get_news_about)
        st.write(scored_data)
    else:
        st.error("Please upload all the required datasets.")
