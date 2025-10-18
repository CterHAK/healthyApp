from load_data import solve_data
from prep_rag_data import create_rag_data
import pandas as pd
from add_enbedding import add_embedding
def main():
    solve_data()
    foods_df = pd.read_csv('../data/processed/foods.csv')
    foods_nutrients = pd.read_csv('../data/processed/food_nutrients.csv')
    rag_df = create_rag_data(foods_df,foods_nutrients)
    add_embedding(rag_df)
if __name__ == '__main__':
    main()
