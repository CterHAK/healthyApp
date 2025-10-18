import pandas as pd
from clean_data import solve_data
import pickle
def create_rag_data(foods_df,nutrients_df):
    foods_df,nutrients_df = solve_data(foods_df,nutrients_df)
    combined_df = pd.merge(foods_df, nutrients_df, on='fdcId', how='left')

    combined_df['rag_text'] = (
        "Food: " + combined_df['description'].fillna('') +
        " (Category: " + combined_df['foodCategory.description'].fillna('Unknown') + ")" +
        " | Nutrient: " + combined_df['nutrient.name'].fillna('Unknown') +
        " | Amount: " + combined_df['amount'].astype(str).replace('0.0', 'N/A').replace('0', 'N/A') +
        " " + combined_df['nutrient.unitName'].fillna('Unknown') +
        " | Data Points: " + combined_df['dataPoints'].astype(str).replace('0', 'N/A') +
        " | Derivation: " + combined_df['foodNutrientDerivation.description'].fillna('Unknown')
    )

    rag_df = combined_df[['fdcId', 'rag_text']].copy()
    rag_df.to_pickle("../data/processed/rag_df.pkl")
    return rag_df


