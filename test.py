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

    total_spend = df[['TV', 'radio', 'newspaper']].sum()
    pie_data = pd.DataFrame({
        'Medium': ['TV', 'Radio', 'Newspaper'],
        'Spend': [total_spend['TV'], total_spend['radio'], total_spend['newspaper']]
    })
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="Spend", type="quantitative"),
        color=alt.Color(field="Medium", type="nominal", legend=alt.Legend(title="Advertising Medium")),
        tooltip=['Medium', 'Spend']
    ).properties(
        title="Advertising Spend Distribution",
        width=400,
        height=400
    )
    st.altair_chart(pie_chart)

    # Perform correlation and regression analysis on numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()

    # Let user select a numeric column to analyze as the target variable
    target_variable = st.selectbox('Select a numeric column to analyze as the target:', numeric_df.columns)

    st.write(f"### Impact Analysis of Variables on {target_variable}")
    
    first_column = True
    col1, col2 = st.columns(2)
    
    for predictor in numeric_df.columns:
        if predictor != target_variable:
            corr_value = correlation.at[target_variable, predictor]
            # Filter to display only strong correlations
            if abs(corr_value) > 0.3:
                # Fit regression model with `predictor` as the independent and `target_variable` as the dependent variable
                X = sm.add_constant(numeric_df[predictor])  # Predictor
                Y = numeric_df[target_variable]  # Response
                model = sm.OLS(Y, X).fit()

                # Check for significant regression results
                if model.pvalues[predictor] < 0.05:
                    # Create a scatter plot using Altair
                    chart = alt.Chart(df).mark_circle(size=60).encode(
                        x=alt.X(predictor, title=predictor),
                        y=alt.Y(target_variable, title=target_variable),
                        tooltip=[predictor, target_variable]
                    ).interactive().properties(
                        width=300,
                        height=300
                    )

                    target_column = first_column and col1 or col2
                    target_column.altair_chart(chart)
                    target_column.write(f"**Impact of {predictor} on {target_variable}:**")
                    target_column.write(f"Correlation : {'increases' if corr_value > 0 else 'decreases'} `{corr_value:.2f}`")
                    target_column.write(f"Every unit increase in `{predictor}` typically results in {model.params[predictor]:.4f} unit {'increase' if model.params[predictor] > 0 else 'decrease'} in `{target_variable}`.")
                    target_column.write(f"This relationship accounts for {model.rsquared:.2%} of the observed variations in `{target_variable}`, indicating a {'strong' if model.rsquared > 0.5 else 'moderate'} influence.")
                    target_column.write(f"The statistical significance of this effect is strong (P-value: {model.pvalues[predictor]:.4g}). This suggests that the changes are likely not due to random fluctuations.")

                    first_column = not first_column

# Check if a file has been uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    display_data(df)
else:
    # Path to the local CSV file
    csv_file_path = 'Advertising.csv'
    df = pd.read_csv(csv_file_path)
    display_data(df)
