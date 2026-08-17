"""
Weaviate Vector Database Adapter — Production Grade

Supports:
- Named vector storage with configurable dimensions
- Rich metadata properties (3GPP section info, spec numbers)
- Filtered vector search with compound conditions
- BM25 keyword search for hybrid retrieval
"""

import weaviate
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.config import Property, DataType, Tokenization, Configure
import logging

logger = logging.getLogger(__name__)

class WeaviateAdapter:
    def __init__(self):
        # Connect to local instance
        self.client = weaviate.connect_to_local()

    def close(self):
        if self.client:
            self.client.close()
            logger.info("Weaviate connection closed.")

    def _clean_name(self, name: str) -> str:
        return "".join(word.capitalize() for word in name.split('_'))

    def upsert(self, collection_name: str, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        class_name = self._clean_name(collection_name)
        
        # Create collection with extended schema if it doesn't exist
        if not self.client.collections.exists(class_name):
            logger.info(f"Creating class '{class_name}' with extended schema...")
            self.client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(), 
                properties=[
                    Property(name="text", data_type=DataType.TEXT),
                    # Filterable IDs (Exact Match)
                    Property(name="user_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="file_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="chunk_num", data_type=DataType.INT),
                    # 3GPP-specific metadata
                    Property(name="section_id", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name="section_title", data_type=DataType.TEXT),
                    Property(name="spec_number", data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                ]
            )

        collection = self.client.collections.get(class_name)

        # Batch Upload
        with collection.batch.dynamic() as batch:
            for i, doc_id in enumerate(ids):
                props = {
                    "text": documents[i],
                    **metadatas[i]
                }
                
                vec = embeddings[i]
                if isinstance(vec, dict):
                    vec = list(vec.values())[0]

                batch.add_object(
                    properties=props,
                    vector=vec, 
                    uuid=self._generate_uuid(doc_id)
                )
        
        # Check for silent failures
        if len(collection.batch.failed_objects) > 0:
            logger.error(f" Failed to upsert {len(collection.batch.failed_objects)} objects!")
            for failed in collection.batch.failed_objects[:3]:
                logger.error(f"   Error: {failed.message}")
        else:
            logger.info(f"Upserted {len(documents)} objects into Weaviate class '{class_name}'")

    def query(self, collection_name: str, query_vector: list[float], top_k: int = 6, where: dict = None):
        """
        Executes a vector similarity search.
        """
        class_name = self._clean_name(collection_name)
        collection = self.client.collections.get(class_name)

        # Construct Filter
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

        # Execute Query
        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=w_filter,
                return_metadata=MetadataQuery(distance=True)
            )
        except Exception as e:
            logger.error(f"Weaviate search failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # Format Results
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

    def keyword_search(self, collection_name: str, query_text: str, top_k: int = 10, where: dict = None):
        """
        BM25 keyword search — critical for exact-match 3GPP acronyms
        (AMF, SMF, UPF, gNB, etc.) that vector search may miss.
        """
        class_name = self._clean_name(collection_name)
        collection = self.client.collections.get(class_name)
        
        # Construct Filter
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
                logger.error(f"BM25 filter construction failed: {e}")
        
        try:
            response = collection.query.bm25(
                query=query_text,
                limit=top_k,
                filters=w_filter,
                query_properties=["text"],
                return_metadata=MetadataQuery(score=True)
            )
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "scores": [[]]}
        
        results = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "scores": [[]]
        }
        
        if response.objects:
            for obj in response.objects:
                results["ids"][0].append(str(obj.uuid))
                results["documents"][0].append(obj.properties.get("text", ""))
                meta = {k: v for k, v in obj.properties.items() if k != "text"}
                results["metadatas"][0].append(meta)
                # BM25 score
                score = obj.metadata.score if obj.metadata.score else 0.0
                results["scores"][0].append(score)
        
        return results

    def _generate_uuid(self, unique_str: str):
        import uuid
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))

db_client = WeaviateAdapter()