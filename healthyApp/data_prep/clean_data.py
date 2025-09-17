from idlelib.replace import replace

import pandas as pd
def holding_column_for_rag(foods_df,nutrients_df):
    foods_selected_columns = [
        'fdcId',
        'description',
        'foodCategory.description'
    ]
    foods_df = foods_df[foods_selected_columns]

    nutrients_selected_columns = [
        'fdcId',
        'foodDescription',
        'nutrient.name',
        'amount',
        'nutrient.unitName',
        'min',
        'median',
        'max',
        'dataPoints',
        'foodNutrientDerivation.description'
    ]
    nutrients_df = nutrients_df[nutrients_selected_columns]
    return  foods_df,nutrients_df

def solve_data(foods_df,nutrients_df):
    foods_df,nutrients_df = holding_column_for_rag(foods_df,nutrients_df)
    foods_df['foodCategory.description'] = (
        foods_df['foodCategory.description']
        .fillna('Unknown')
        .str.lower()
    )
    # Numeric columns in nutrients_df
    numeric_cols = ['amount', 'min', 'median', 'max', 'dataPoints']

    # Replace 'N/A' strings with NaN, convert to numeric, then fillna
    nutrients_df[numeric_cols] = (
        nutrients_df[numeric_cols]
        .replace('N/A', pd.NA)
        .apply(pd.to_numeric, errors='coerce')
        .fillna(0)
    )

    nutrients_df['dataPoints'] = nutrients_df['dataPoints'].astype(pd.Int64Dtype())

    text_cols = ['nutrient.name', 'nutrient.unitName', 'foodNutrientDerivation.description']

    nutrients_df[text_cols] = nutrients_df[text_cols].fillna('Unknown')

    nutrients_df['nutrient.name'] = nutrients_df['nutrient.name'].str.lower()


    invalid_amount_count = (nutrients_df['amount'] < 0).sum()
    if invalid_amount_count > 0:
        print(f"Warning: Found {invalid_amount_count} rows with negative nutrient amounts. These are set to 0.")
        nutrients_df.loc[nutrients_df['amount'] < 0, 'amount'] = 0
    return foods_df,nutrients_df
