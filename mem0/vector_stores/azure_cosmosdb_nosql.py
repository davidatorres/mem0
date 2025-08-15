import json
import logging
from pydantic import BaseModel
from mem0.memory.utils import extract_json
from mem0.vector_stores.base import VectorStoreBase
from typing import Optional

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey, ThroughputProperties, exceptions
    from azure.identity import DefaultAzureCredential
except ImportError:
    raise ImportError(
        "The 'azure-search-documents' library is required. Please install it using 'pip install azure-search-documents==11.5.2'."
    )

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]


class AzureCosmosDbNoSql(VectorStoreBase):
    def __init__(
        self,
        service_name: str,
        database_name: str,
        container_name: str,
        api_key: str,
        auto_scale: bool = False,
        max_throughput: int = 400,
        vector_size: int = 1536,
        distance: str = "cosine",
    ) -> None:
        self.service_name = service_name
        self.container_name = container_name
        self.database_name = database_name
        self.api_key = api_key
        self.auto_scale = auto_scale
        self.max_throughput = max_throughput
        self.vector_size = vector_size
        self.distance = distance
        self.endpoint = f"https://{self.service_name}.documents.azure.com:443/"

        # If the API key is not provided or is a placeholder, use DefaultAzureCredential.
        if self.api_key is None or self.api_key == "" or self.api_key == "your-api-key":
            self.credential = DefaultAzureCredential()
            self.api_key = None
        else:
            self.credential = self.api_key

        self._check_database()
        # self._create_col_if_not_exist()

    def _check_database(self):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            databases = [db["id"] for db in client.list_databases()]
            if self.database_name not in databases:
                try:
                    database = client.create_database(
                        id=self.database_name,
                        offer_throughput=self.max_throughput
                    )
                    logger.info(f"Created database: {self.database_name}")
                    return False, 400
                except exceptions.CosmosHttpResponseError as e:
                    if "Shared throughput database creation is not supported for serverless accounts" in str(e):
                        logger.warning("Creating a vector store is not supported on Cosmos DB serverless accounts")
                        return False, -1
            else:
                database = client.get_database_client(self.database_name)
                try:
                    throughput = database.get_throughput()
                    is_autoscale = True if throughput.auto_scale_max_throughput else False
                    throughput_max = throughput.auto_scale_max_throughput if is_autoscale else throughput.offer_throughput
                    logger.info(
                        f"Database: {database.id}, Autoscale Max: {throughput.auto_scale_max_throughput}, "
                        f"Manual Max: {throughput.offer_throughput}",
                    )
                    return is_autoscale, throughput_max
                except exceptions.CosmosHttpResponseError as e:
                    if "Reading or replacing offers is not supported for serverless accounts" in str(e):
                        logger.warning("Creating a vector store is not supported on Cosmos DB serverless accounts")
                        return False, -1
                except AttributeError:
                    logger.warning(f"Database {database.id} does not have throughput defined, throughput must be set at creation time.")
                    return False, -1
        
        # Default return if no conditions are met
        return False, -1

    def _create_col_if_not_exist(self):
        try:
            with CosmosClient(url=self.endpoint, credential=self.credential) as client:
                database = client.get_database_client(self.database_name)
                try:
                    containers = [c["id"] for c in database.list_containers()]
                    if self.container_name not in containers:
                        self.create_col(self.container_name, vector_size=self.vector_size, distance=self.distance)
                except exceptions.CosmosHttpResponseError as e:
                    logger.error(f"Failed to list containers: {e}")
                    raise (e)
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"Failed to create container: {e}")
            raise (e)

    def _generate_document(self, vector, payload, id):
        document = {"id": id, "vector": vector, "payload": json.dumps(payload)}
        # Extract additional fields if they exist.
        for field in ["user_id", "run_id", "agent_id"]:
            if field in payload:
                document[field] = payload[field]
        return document

    def insert(self, vectors, payloads=None, ids=None):
        logger.info(f"Inserting {len(vectors)} vectors into container {self.container_name}.")
        documents = [
            self._generate_document(vector, payload, id) for id, vector, payload in zip(ids, vectors, payloads)
        ]
        results = []
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            for document in documents:
                try:
                    result = container.upsert_item(body=document)
                    results.append(result)
                except exceptions.CosmosHttpResponseError as e:
                    logger.error(f"Failed to insert document with id {document.get('id')}: {e}")
                    results.append(None)

    def search(self, query, vectors, limit=5, filters=None):
        results = []

        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            search_results = list(
                container.query_items(
                    query=query,
                    max_item_count=limit,
                    enable_cross_partition_query=True,
                )
            )
            for result in search_results:
                payload = json.loads(extract_json(result["payload"]))
                results.append(OutputData(id=result["id"], score=0, payload=payload))

        return results

    def delete(self, vector_id):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)

            try:
                container.delete_item(item=vector_id, partition_key=vector_id)
            except exceptions.CosmosHttpResponseError as e:
                if e.status_code == 404:
                    logger.warning(f"Document with id {vector_id} not found.")

        return None

    def update(self, vector_id, vector=None, payload=None):
        document = {"id": vector_id}
        if vector:
            document["vector"] = vector
        if payload:
            json_payload = json.dumps(payload)
            document["payload"] = json_payload
            for field in ["user_id", "run_id", "agent_id"]:
                document[field] = payload.get(field)

        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            container.upsert_item(body=document)

        return None

    def get(self, vector_id) -> OutputData | None:
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)
            document = container.read_item(item=vector_id, partition_key=vector_id)
            payload = json.loads(extract_json(document["payload"]))
            return OutputData(id=document["id"], score=None, payload=payload)

    def list_cols(self):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            containers = database.list_containers()
        return list(containers)

    def delete_col(self):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            database.delete_container(self.container_name)

    def col_info(self):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)

        container_properties = container.read()
        info = {
            "id": container_properties.get("id"),
            "resource_id": container_properties.get("_rid"),
            "etag": container_properties.get("_etag"),
            "last_modified": container_properties.get("_ts"),
            "partition_key": container_properties.get("partitionKey"),
            "indexing_policy": container_properties.get("indexingPolicy"),
            "default_ttl": container_properties.get("defaultTtl"),
            "throughput": None,
        }

        return info

    def list(self, filters=None, limit=None):
        results = []

        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            container = database.get_container_client(self.container_name)

            documents = list(container.read_all_items())
            for document in documents:
                payload = None
                if "payload" in document:
                    try:
                        payload = json.loads(extract_json(document["payload"]))
                    except Exception:
                        payload = document["payload"]
                    results.append(OutputData(id=document.get("id"), score=None, payload=payload))

        return results

    def reset(self):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            database.delete_container(self.container_name)

        # Provide appropriate values for name, vector_size, and distance
        self.create_col(self.container_name, vector_size=self.vector_size, distance=self.distance)

    def create_col(self, name, vector_size=1536, distance="cosine"):
        # Check the database first
        is_autoscale, throughput_max = self._check_database()
        if throughput_max == -1:
            logger.warning(f"Database {self.database_name} does not support vector stores.")
            return

        # Define the vector embedding policy
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {"path": "/vector", "dataType": "float32", "distanceFunction": distance, "dimensions": vector_size}
            ]
        }

        # Define the full-text search policy
        full_text_policy = {"defaultLanguage": "en-US", "fullTextPaths": [{"path": "/payload", "language": "en-US"}]}

        # Define the indexing policy with both vector and full-text indexes
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/_etag/?"}, {"path": "/vector/*"}],
            "fullTextIndexes": [{"path": "/payload"}],
            "vectorIndexes": [
                {"path": "/vector", "type": "diskANN", "quantizationByteSize": 96, "indexingSearchListSize": 100}
            ],
        }

        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)

            if is_autoscale:
                throughput_properties = ThroughputProperties(auto_scale_max_throughput=1000)
            else:
                throughput_properties = ThroughputProperties(offer_throughput=1000)

            try:
                container = database.create_container_if_not_exists(
                    id=name,
                    partition_key=PartitionKey(path="/id"),
                    indexing_policy=indexing_policy,
                    vector_embedding_policy=vector_embedding_policy,
                    full_text_policy=full_text_policy,
                    offer_throughput=throughput_properties,
                )
            except exceptions.CosmosHttpResponseError as e:
                logger.error(f"Error creating container '{name}': {e}")
                raise (e)
            except Exception as e:
                msg = str(e)
                logger.error(f"Unexpected error creating container '{name}': {e}")
                raise (e)


# "A Container Vector Policy has been provided, but the capability has not been enabled on your account."
