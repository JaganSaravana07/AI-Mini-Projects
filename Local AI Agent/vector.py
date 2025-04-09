from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
'''
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)'''
# Load dataset
df = pd.read_csv("restaurant_customer_satisfaction.csv")

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        content = (
            f"Visit Frequency: {row['VisitFrequency']}, "
            f"Preferred Cuisine: {row['PreferredCuisine']}, "
            f"Dining Occasion: {row['DiningOccasion']}, "
            f"Meal Type: {row['MealType']}, "
            f"Service Rating: {row['ServiceRating']}, "
            f"Food Rating: {row['FoodRating']}, "
            f"Ambiance Rating: {row['AmbianceRating']}, "
            f"High Satisfaction: {row['HighSatisfaction']}"
        )
        
        document = Document(
            page_content=content,
            metadata={"customer_id": row["CustomerID"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

    # Create and persist vector store
    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)

# Initialize retriever
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})