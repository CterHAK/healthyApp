from sentence_transformers import SentenceTransformer
import pickle
def add_embedding(rag_df):
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    # Generate embeddings for the 'rag_text' column
    print("Generating embeddings...")
    rag_df['embeddings'] = rag_df['rag_text'].apply(lambda x: model.encode(x))
    print("Embeddings generated successfully.")

    rag_df.to_pickle('../data/processed/rag_df.pkl')
