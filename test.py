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

    # Perform correlation and regression analysis on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()

    # Let user select a numeric column to analyze
    column_to_analyze = st.selectbox('Select a numeric column to analyze:', numeric_df.columns)

    st.write(f"### Impact Analysis for {column_to_analyze}")
    selected_correlation = correlation[column_to_analyze].drop(column_to_analyze)  # Remove self-correlation

    first_column = True
    col1, col2 = st.columns(2)
    
    for col in selected_correlation.index:
        if abs(selected_correlation[col]) > 0.5:
            # Fit regression model
            X = sm.add_constant(numeric_df[column_to_analyze])  # Predictor
            Y = numeric_df[col]  # Response
            model = sm.OLS(Y, X).fit()

            # Check for significant correlation and regression
            if model.pvalues[column_to_analyze] < 0.05:
                # Create a scatter plot using Altair
                chart = alt.Chart(df).mark_circle(size=60).encode(
                    x=alt.X(column_to_analyze, title=column_to_analyze),
                    y=alt.Y(col, title=col),
                    tooltip=[column_to_analyze, col]
                ).interactive().properties(
                    width=300,
                    height=300
                )

                target_column = first_column and col1 or col2
                target_column.altair_chart(chart)
                target_column.write(f"**Impact of changing {column_to_analyze} on {col}:**")
                target_column.write(f"Correlation impact: {'increases' if selected_correlation[col] > 0 else 'decreases'}")
                target_column.write(f"Every unit increase in `{column_to_analyze}` typically results in {model.params[column_to_analyze]:.4f} unit {'increase' if model.params[column_to_analyze] > 0 else 'decrease'} in `{col}`.")
                target_column.write(f"This relationship accounts for {model.rsquared:.2%} of the changes observed in `{col}`, indicating a {'strong' if model.rsquared > 0.5 else 'moderate'} influence.")
                target_column.write(f"The statistical significance of this effect is strong (P-value: {model.pvalues[column_to_analyze]:.4g}). This suggests that the changes are likely not due to random fluctuations.")

                first_column = not first_column

# Check if a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    display_data(df)
else:
    # Path to the local CSV file
    csv_file_path = 'online_shoppers_intention.csv'
    df = pd.read_csv(csv_file_path)
    display_data(df)
