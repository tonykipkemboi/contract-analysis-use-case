import os
from qdrant_client import QdrantClient
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import boto3
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import uuid
import json

load_dotenv()

def get_indexed_files(client: QdrantClient, collection_name: str) -> set:
    """Get list of already indexed files from collection metadata."""
    try:
        # Search for all points and get unique filenames from metadata
        results = client.scroll(
            collection_name=collection_name,
            limit=10000,  # Adjust based on your needs
            with_payload=True,
            with_vectors=False,
        )[0]  # Get first element of the tuple which contains the points
        
        indexed_files = set()
        for point in results:
            if point.payload and point.payload.get("metadata"):
                filename = point.payload["metadata"].get("origin", {}).get("filename")
                if filename:
                    indexed_files.add(filename)
        
        print(f"Found {len(indexed_files)} previously indexed files")
        return indexed_files
    except Exception as e:
        print(f"Error getting indexed files: {e}")
        return set()

def pre_process_docs(force_reindex: bool = False):
    """Process PDF documents and store them in Qdrant.
    
    Args:
        force_reindex (bool): If True, reindex all files regardless of whether they've been indexed before.
    """
    # Setup clients
    COLLECTION_NAME = "contracts_business_5"
    doc_converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Initialize Bedrock client
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION_NAME"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedding_model = "amazon.titan-embed-g1-text-02"

    # Get list of already indexed files
    indexed_files = set() if force_reindex else get_indexed_files(client, COLLECTION_NAME)
    
    def get_embedding(text: str) -> list[float]:
        """Generate embeddings using AWS Bedrock's Titan model."""
        request_body = {
            "inputText": text
        }
        response = bedrock.invoke_model(
            modelId=embedding_model,
            body=json.dumps(request_body)
        )
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')

    # Define the folder where PDFs are stored
    pdf_folder = "knowledge/contracts/"
    points = []
    new_files = []

    # Loop through all PDFs in the folder and process them
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            if filename in indexed_files and not force_reindex:
                print(f"Skipping {filename} - already indexed")
                continue
                
            new_files.append(filename)
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing {pdf_path}")

            result = doc_converter.convert(pdf_path)

            # Chunk the converted document
            for chunk in HybridChunker().chunk(result.document):
                print("chunk", chunk)
                vector = get_embedding(chunk.text)
                point_id = str(uuid.uuid4())
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "text": chunk.text,
                            "metadata": chunk.meta.export_json_dict(),
                        },
                    )
                )

    if not points:
        print("No new files to process")
        return

    print(f"New files to be indexed: {new_files}")
    print(f"Total new points to be added: {len(points)}")

    # Create collection if it doesn't exist
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"Collection exists with {collection_info.points_count} points")
    except:
        print("Creating new collection")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Add new points to collection
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Successfully processed and stored {len(points)} new chunks in Qdrant")

if __name__ == "__main__":
    pre_process_docs()
