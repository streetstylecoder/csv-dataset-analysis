import streamlit as st
import pandas as pd
import statsmodels.api as sm
import altair as alt
import numpy as np

# Title of the application
st.title('CSV Dataset Analysis')

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to display data, descriptions, correlations, perform regression, and plot scatter charts
def display_data(df):
    st.write("### Data Preview", df.head())

    st.write("""
    ### Column Descriptions:
    - **Administrative**: Number of pages of this type (administrative) that the user visited.
    - **Administrative_Duration**: Amount of time spent in this category of pages.
    - **Informational**: Number of pages of this type (informational) that the user visited.
    - **Informational_Duration**: Amount of time spent in this category of pages.
    - **ProductRelated**: Number of pages of this type (product-related) that the user visited.
    - **ProductRelated_Duration**: Amount of time spent in this category of pages.
    - **BounceRates**: Percentage of visitors who enter the website through that page and exit without triggering any additional tasks.
    - **ExitRates**: Percentage of pageviews on the website that end at that specific page.
    - **PageValues**: Average value of the page averaged over the value of the target page and/or the completion of an eCommerce transaction.
    - **SpecialDay**: This value represents the closeness of the browsing date to special days or holidays (e.g., Mother's Day or Valentine's Day).
    """)

    # Perform correlation and regression analysis on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()
    st.write("### Correlation Matrix")
    st.dataframe(correlation.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

    st.write("### Relationship Analysis and Scatter Plots")
    first_column = True
    col1, col2 = st.columns(2)
    
    for col1_name in numeric_df.columns:
        for col2_name in numeric_df.columns:
            if col1_name != col2_name:
                # Fit regression model
                X = sm.add_constant(numeric_df[col1_name])  # Predictor
                Y = numeric_df[col2_name]  # Response
                model = sm.OLS(Y, X).fit()

                # Check for significant correlation and regression
                if abs(correlation.at[col1_name, col2_name]) > 0.5 and model.pvalues[col1_name] < 0.05:
                    # Create a scatter plot using Altair
                    chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X(col1_name, title=col1_name),
                        y=alt.Y(col2_name, title=col2_name),
                        tooltip=[col1_name, col2_name]
                    ).interactive().properties(
                        width=300,
                        height=300
                    )

                    if first_column:
                        col1.altair_chart(chart)
                        col1.write(f"Correlation coefficient: {correlation.at[col1_name, col2_name]:.4f}")
                        col1.write(f"Regression coefficient for `{col1_name}`: {model.params[col1_name]:.4f}")
                        col1.write(f"R-squared: {model.rsquared:.4f}")
                        col1.write(f"P-value: {model.pvalues[col1_name]:.4g}")
                        first_column = False
                    else:
                        col2.altair_chart(chart)
                        col2.write(f"Correlation coefficient: {correlation.at[col1_name, col2_name]:.4f}")
                        col2.write(f"Regression coefficient for `{col1_name}`: {model.params[col1_name]:.4f}")
                        col2.write(f"R-squared: {model.rsquared:.4f}")
                        col2.write(f"P-value: {model.pvalues[col1_name]:.4g}")
                        first_column = True

# Check if a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    display_data(df)
else:
    # Path to the local CSV file
    csv_file_path = 'online_shoppers_intention.csv'
    df = pd.read_csv(csv_file_path)
    display_data(df)
