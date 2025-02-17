import json
import os
from typing import Any, Optional, Type
import boto3


try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = Any  # type placeholder
    Filter = Any
    FieldCondition = Any
    MatchValue = Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QdrantToolSchema(BaseModel):
    """Input for QdrantTool."""

    query: str = Field(
        ...,
        description="The query to search retrieve relevant information from the Qdrant database. Pass only the query, not the question.",
    )
    filter_by: Optional[str] = Field(
        default=None,
        description="Filter by properties. Pass only the properties, not the question.",
    )
    filter_value: Optional[str] = Field(
        default=None,
        description="Filter by value. Pass only the value, not the question.",
    )


class QdrantVectorSearchTool(BaseTool):
    """Tool to query, and if needed filter results from a Qdrant database"""

    model_config = {"arbitrary_types_allowed": True}
    client: QdrantClient = None
    bedrock: Any = None  # Add bedrock client field
    name: str = "QdrantVectorSearchTool"
    description: str = "A tool to search the Qdrant database for relevant information on internal documents."
    args_schema: Type[BaseModel] = QdrantToolSchema
    query: Optional[str] = None
    filter_by: Optional[str] = None
    filter_value: Optional[str] = None
    collection_name: Optional[str] = None
    limit: Optional[int] = Field(default=3)
    score_threshold: float = Field(default=0.35)
    qdrant_url: str = Field(
        ...,
        description="The URL of the Qdrant server",
    )
    qdrant_api_key: str = Field(
        ...,
        description="The API key for the Qdrant server",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if QDRANT_AVAILABLE:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
            )
            # Initialize Bedrock client
            self.bedrock = boto3.client(
                service_name="bedrock-runtime",
                region_name=os.getenv("AWS_REGION_NAME"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )

    def _run(
        self,
        query: str,
        filter_by: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> str:
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' package is required to use the QdrantVectorSearchTool. "
                "Please install it with: pip install qdrant-client"
            )

        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set")

        # Create filter if filter parameters are provided
        search_filter = None
        if filter_by and filter_value:
            search_filter = Filter(
                must=[
                    FieldCondition(key=filter_by, match=MatchValue(value=filter_value))
                ]
            )

        # Search in Qdrant using the built-in query method
        query_vector = self.vectorize_query(query)
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=self.limit,
            score_threshold=self.score_threshold,
        )

        # Format results similar to storage implementation
        results = []
        # Extract the list of ScoredPoint objects from the tuple
        for point in search_results:
            result = {
                "metadata": point[1][0].payload.get("metadata", {}),
                "context": point[1][0].payload.get("text", ""),
                "distance": point[1][0].score,
            }
            results.append(result)

        return json.dumps(results, indent=2)

    def vectorize_query(self, query: str) -> list[float]:
        """Generate embeddings using AWS Bedrock's Titan model."""
        import json

        # Prepare the request body
        request_body = {
            "inputText": query
        }

        # Call Bedrock's embedding endpoint
        response = self.bedrock.invoke_model(
            modelId="amazon.titan-embed-g1-text-02",
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        embedding = response_body.get('embedding')
        
        return embedding


# if __name__ == "__main__":
#     tool = QdrantVectorSearchTool(
#         collection_name="contracts_business_5",
#         qdrant_url=os.getenv("QDRANT_URL"),
#         qdrant_api_key=os.getenv("QDRANT_API_KEY"),
#     )
#     print(tool.run("What is the grants to rights of digital cinema destinations corp?"))
