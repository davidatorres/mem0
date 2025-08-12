import json
import logging
from pydantic import BaseModel
from mem0.memory.utils import extract_json
from mem0.vector_stores.base import VectorStoreBase
from typing import Optional

try:
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
        max_throughput: int = 400,
        vector_size: int = 1536,
        distance: str = "cosine"
    ) -> None:
        self.service_name = service_name
        self.container_name = container_name
        self.database_name = database_name
        self.api_key = api_key
        self.max_throughput = max_throughput
        self.vector_size = vector_size
        self.distance = distance
        self.endpoint = f"https://{self.service_name}.documents.azure.com:443/"
        
        # If the API key is not provided or is a placeholder, use DefaultAzureCredential.
        if self.api_key is None or self.api_key == "" or self.api_key == "your-api-key":
            self.credential = DefaultAzureCredential()
        else:
            self.credential = self.api_key

    def create_col(self, name, vector_size, distance):
        with CosmosClient(url=self.endpoint, credential=self.credential) as client:
            database = client.get_database_client(self.database_name)
            
            # Check if the database is set to autoscale by inspecting its throughput properties
            throughput: ThroughputProperties = database.read_offer()
            is_autoscale = False
            if throughput and hasattr(throughput, "offer") and "autoscaleSettings" in throughput["offer"]:
                is_autoscale = True
            logger.info(f"Database autoscale enabled: {is_autoscale}")
            
            container = database.create_container_if_not_exists(
                id=name,
                partition_key=PartitionKey(path="/id"),
                offer_throughput=ThroughputProperties(
                    autoscale=is_autoscale,
                    max_throughput=self.max_throughput,
                )
            )

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
        pass
    
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
            



    def create_container(
        self,
        container_name: str,
        host_uri: str | None = None,
        database_name: str | None = None,
        indexing_policy: dict = {},
        vector_embedding_policy: dict = {},
        full_text_policy: dict = {},
        offer_throughput: int = 0,
    ) -> ContainerProxy | None:
        """
        Create a container if it does not exist in the specified database.

        Parameters:
            container_name (str): Name for the new container.
            host_uri (str | None): Optional endpoint override.
            database_name (str | None): Name of the database.
            indexing_policy (dict): Indexing policy settings.
            vector_embedding_policy (dict, optional): The vector embedding policy for the container. Defaults to None.
            full_text_policy (dict, optional): The full text policy for the container. Defaults to None.
            offer_throughput (int, optional): The throughput for the container. Defaults to 400.

        Returns:
            ContainerProxy | None: The created or existing container proxy.
        """
        host_uri = host_uri if host_uri else self.properties.host_uri
        database_name = database_name if database_name else self.properties.database_name

        if host_uri == self.properties.host_uri and database_name == self.properties.database_name:
            if self.database:
                database = self.database
            else:
                database = self.get_database(host_uri, database_name=database_name)

        # Create throughput property based on the offer_throughput value
        if offer_throughput < 0:
            # Autoscale throughput
            if offer_throughput > -1000:  # -1 to -999
                offer_throughput_property = ThroughputProperties(auto_scale_max_throughput=1000)
            elif offer_throughput > -10000:  # -1000 to -9999
                offer_throughput_property = ThroughputProperties(auto_scale_max_throughput=abs(offer_throughput))
            else:
                raise ValueError("Offer throughput must be between -1 and 10000. where negative values autoscale.")
        else:
            # Manual throughput
            if offer_throughput < 400:  # 1 to 399
                offer_throughput_property = ThroughputProperties(offer_throughput=400)
            elif offer_throughput < 10001:  # 400 to 10000
                offer_throughput_property = ThroughputProperties(offer_throughput=offer_throughput)
            else:
                raise ValueError("Offer throughput must be between -1 and 10000. where -1 is autoscale.")

        if database:
            return database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/id"),
                indexing_policy=indexing_policy,
                vector_embedding_policy=vector_embedding_policy,
                full_text_policy=full_text_policy,
                offer_throughput=offer_throughput_property,
            )
        else:
            self.logger.warning(f"Database with name {database_name} not found.")
            return None



    def query_documents(
        self,
        query: str,
        host_uri: str | None = None,
        database_name: str | None = None,
        container_name: str | None = None,
    ) -> list | None:
        """
        Execute a query on the container and return matching documents.

        Parameters:
            query (str): The SQL-like query string.
            host_uri (str | None): Optional endpoint override.
            database_name (str | None): Name of the database.
            container_name (str | None): Name of the container.

        Returns:
            list | None: A list of documents that satisfy the query or None if not found.

        Raises:
            ApplicationException: If the query is missing or execution fails.
        """
        if query is None:
            raise ApplicationException("Query must be provided.")
        try:
            container = self.get_container(host_uri, database_name, container_name)
            if container:
                return list(container.query_items(query=query, enable_cross_partition_query=True))
            else:
                self.logger.warning(f"Container with name {container_name} not found.")
                return None
        except Exception as e:
            raise ApplicationException(f"Exception in query_container: {str(e)}")


