import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()
COLLECTION_NAME = "indian_laws"

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

# Create indexes for filtering
print("Creating index on 'category'...")
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

print("Creating index on 'jurisdiction'...")
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="jurisdiction",
    field_schema=PayloadSchemaType.KEYWORD
)

print("Indexes created. You can now filter queries.")