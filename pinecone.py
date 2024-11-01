from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="chave_api-pinecone")

index_name = "quickstart"

pc.create_index(
    name=index_name,
    dimension=8, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
