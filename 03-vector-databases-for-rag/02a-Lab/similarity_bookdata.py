# Importing the necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions
import json

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Define the name for the collection to be created or retrieved
collection_name = "book_collection"

# Define the main function to interact with the Chroma DB
def main():
    try:
        # Create a collection in the Chroma database with a specified name, 
        # distance metric, and embedding function. In this case, we are using 
        # cosine distance
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "A collection for storing book data"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        print(f"Collection created: {collection.name}")

        # List of book dictionaries with comprehensive details for advanced search
        books = json.load(open("books.json", "r"))

        # Create comprehensive text documents for each book
        book_documents = []
        for book in books:
            document = f"{book['title']} by {book['author']}. {book['description']} "
            document += f"Themes: {book['themes']}. Setting: {book['setting']}. "
            document += f"Genre: {book['genre']} published in {book['year']}."
            book_documents.append(document)

        # Adding book data to the collection with comprehensive metadata
        collection.add(
            ids=[book["id"] for book in books],
            documents=book_documents,
            metadatas=[{
                "title": book["title"],
                "author": book["author"],
                "genre": book["genre"],
                "year": book["year"],
                "rating": book["rating"],
                "pages": book["pages"]
            } for book in books]
        )

        # Retrieve all the items (documents) stored in the collection
        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform advanced book search
        def perform_book_search(collection):
            print("=== Book Similarity Search ===")
            
            # Similarity search for magical adventures
            print("\n1. Finding magical fantasy adventures:")
            results = collection.query(
                query_texts=["magical fantasy adventure with friendship and courage"],
                n_results=3
            )
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
            )):
                metadata = results['metadatas'][0][i]
                print(f"  {i+1}. {metadata['title']} by {metadata['author']} - Distance: {distance:.4f}")
            
            print("\n=== Metadata Filtering ===")
            
            # Filter by genre
            print("\n2. Finding Fantasy and Science Fiction books:")
            results = collection.get(
                where={"genre": {"$in": ["Fantasy", "Science Fiction"]}}
            )
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                print(f"  - {metadata['title']}: {metadata['genre']} ({metadata['rating']}★)")
            
            # Filter by rating
            print("\n3. Finding highly-rated books (4.3+):")
            results = collection.get(
                where={"rating": {"$gte": 4.3}}
            )
            for i, doc_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                print(f"  - {metadata['title']}: {metadata['rating']}★")
            
            print("\n=== Combined Search ===")
            
            # Combined search: dystopian themes with high ratings
            print("\n4. Finding highly-rated dystopian books:")
            results = collection.query(
                query_texts=["dystopian society control oppression future"],
                n_results=3,
                where={"rating": {"$gte": 4.0}}
            )
            for i, (doc_id, document, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['distances'][0]
            )):
                metadata = results['metadatas'][0][i]
                print(f"  {i+1}. {metadata['title']} ({metadata['year']}) - {metadata['rating']}★")
                print(f"     Distance: {distance:.4f}")

        perform_book_search(collection)
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    main()