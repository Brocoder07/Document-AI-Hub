import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.config import Property, DataType, Tokenization, Configure
import os
import logging
logger = logging.getLogger(__name__)

class WeaviateAdapter:
    def __init__(self):
        self.client = weaviate.connect_to_local(
            headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")}
        )

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed.")

    def _clean_name(self, name: str) -> str:
        return "".join(word.capitalize() for word in name.split('_'))

    def upsert(self, collection_name: str, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        class_name = self._clean_name(collection_name)
        
        # 1. Define Schema with EXPLICIT Named Vector
        if not self.client.collections.exists(class_name):
            logger.info(f"Creating class '{class_name}' with explicit schema...")
            self.client.collections.create(
                name=class_name,
                # Explicitly configure the 'default' vector space
                vector_config=[
                    Configure.NamedVectors.none(name="default")
                ],
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    # IDs as Keys (Exact Match)
                    Property(name="user_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="file_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="chunk_num", data_type=DataType.INT),
                ]
            )

        collection = self.client.collections.get(class_name)

        # 2. Batch Upload with Error Checking
        with collection.batch.dynamic() as batch:
            for i, doc_id in enumerate(ids):
                props = {
                    "text": documents[i],
                    **metadatas[i]
                }
                
                # Ensure vector is named 'default'
                vec = embeddings[i]
                if not isinstance(vec, dict):
                    vec = {"default": vec}

                batch.add_object(
                    properties=props,
                    vector=vec, 
                    uuid=self._generate_uuid(doc_id)
                )
        
        # 3. CRITICAL: Check for silent failures
        if len(collection.batch.failed_objects) > 0:
            logger.error(f"❌ Failed to upsert {len(collection.batch.failed_objects)} objects!")
            for failed in collection.batch.failed_objects[:3]:
                logger.error(f"   Error: {failed.message}")
        else:
            logger.info(f"Upserted {len(documents)} objects into Weaviate class '{class_name}'")

    def query(self, collection_name: str, query_vector: list[float], top_k: int = 6, where: dict = None):
        class_name = self._clean_name(collection_name)
        collection = self.client.collections.get(class_name)

        w_filter = None
        if where:
            try:
                if "$and" in where:
                    conditions = where["$and"]
                    if conditions:
                        k, v = list(conditions[0].items())[0]
                        w_filter = Filter.by_property(k).equal(v)
                        for cond in conditions[1:]:
                            k, v = list(cond.items())[0]
                            w_filter = w_filter & Filter.by_property(k).equal(v)
                else:
                    k, v = list(where.items())[0]
                    w_filter = Filter.by_property(k).equal(v)
            except Exception as e:
                logger.error(f"Filter construction failed: {e}")

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=w_filter,
                return_metadata=MetadataQuery(distance=True),
                target_vector="default" # Must match the schema name
            )
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        if response.objects:
            for obj in response.objects:
                results["ids"][0].append(str(obj.uuid))
                results["documents"][0].append(obj.properties.get("text", ""))
                meta = {k: v for k, v in obj.properties.items() if k != "text"}
                results["metadatas"][0].append(meta)
                results["distances"][0].append(obj.metadata.distance)

        return results

    def _generate_uuid(self, unique_str: str):
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

db_client = WeaviateAdapter()