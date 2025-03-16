import os
import time
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

logger = logging.getLogger(__name__)

def delete_existing_pdfs():
    try:
        blob_service = BlobServiceClient.from_connection_string(os.environ.get("AZURE_STORAGE_CONNECTION_STRING"))
        container_client = blob_service.get_container_client(os.environ.get("CONTAINER_NAME"))

        blobs = container_client.list_blobs()
        for blob in blobs:
            container_client.delete_blob(blob.name)
        return True
    except Exception as e:
        logger.error(f"Error deleting PDFs: {str(e)}")
        return False

def create_search_resources():
    try:
        import requests
        
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
        DATASOURCE_NAME = os.environ.get("DATASOURCE_NAME", "simplerag")
        INDEX_NAME = os.environ.get("INDEX_NAME", "azureblob-index")
        INDEXER_NAME = os.environ.get("INDEXER_NAME", "azureblob-indexer")
        AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
        
        # Common headers for all requests
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_KEY
        }
        
        # 1. Create data source if not exists
        data_source_url = f"{AZURE_SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version=2020-06-30"
        data_source_body = {
            "name": DATASOURCE_NAME,
            "type": "azureblob",
            "credentials": {
                "connectionString": AZURE_STORAGE_CONNECTION_STRING
            },
            "container": {
                "name": CONTAINER_NAME
            }
        }
        
        data_source_created = False
        try:
            ds_response = requests.put(data_source_url, headers=headers, json=data_source_body)
            ds_response.raise_for_status()
            print(f"Data source '{DATASOURCE_NAME}' created or updated")
            data_source_created = True
        except Exception as e:
            print(f"Error creating data source: {str(e)}")
            return False
        
        # 2. Create index if not exists
        index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
        index_body = {
            "name": INDEX_NAME,
            "fields": [
                {
                    "name": "id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True
                },
                {
                    "name": "content",
                    "type": "Edm.String",
                    "searchable": True,
                    "filterable": False,
                    "sortable": False,
                    "facetable": False,
                    "retrievable": True
                },
                {
                    "name": "metadata_storage_name",
                    "type": "Edm.String",
                    "searchable": False,
                    "retrievable": True
                },
                {
                    "name": "metadata_storage_path",
                    "type": "Edm.String",
                    "searchable": False,
                    "retrievable": True
                }
            ]
        }
        
        index_created = False
        try:
            idx_response = requests.put(index_url, headers=headers, json=index_body)
            idx_response.raise_for_status()
            print(f"Index '{INDEX_NAME}' created or updated")
            index_created = True
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return False
        
        # 3. Create indexer if not exists
        indexer_url = f"{AZURE_SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version=2020-06-30"
        indexer_body = {
            "name": INDEXER_NAME,
            "dataSourceName": DATASOURCE_NAME,
            "targetIndexName": INDEX_NAME,
            "parameters": {
                "configuration": {
                    "parsingMode": "default",
                    "dataToExtract": "contentAndMetadata"
                }
            }
        }
        
        indexer_created = False
        try:
            idx_response = requests.put(indexer_url, headers=headers, json=indexer_body)
            idx_response.raise_for_status()
            print(f"Indexer '{INDEXER_NAME}' created or updated")
            indexer_created = True
            # Small delay to ensure registration
            time.sleep(2)
        except Exception as e:
            print(f"Error creating indexer: {str(e)}")
            return False
        
        # Run the indexer immediately
        if indexer_created:
            try:
                run_url = f"{AZURE_SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}/run?api-version=2020-06-30"
                run_response = requests.post(run_url, headers=headers)
                run_response.raise_for_status()
                print(f"Indexer '{INDEXER_NAME}' triggered to run")
            except Exception as e:
                print(f"Error running indexer: {str(e)}, but resources were created")
                # Don't fail here as resources are created, just indexer run failed
        
        # If we've gotten this far, all resources are created
        return data_source_created and index_created and indexer_created
        
    except Exception as e:
        print(f"Error creating search resources: {str(e)}")
        return False

def reset_and_run_indexer():
    try:
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
        INDEXER_NAME = os.environ.get("INDEXER_NAME", "azureblob-indexer")
        
        # 인덱서 클라이언트 생성
        indexer_client = SearchIndexerClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        # 필요한 리소스 생성 확인
        try:
            # Use the correct indexer name as defined in our constants
            indexer_client.get_indexer(INDEXER_NAME)
        except ResourceNotFoundError:
            # 인덱서가 없으면 필요한 리소스 생성
            create_search_resources()
        
        # 인덱서를 리셋(기존 데이터 삭제 후 다시 시작)
        indexer_client.reset_indexer(INDEXER_NAME)
        # 인덱서 즉시 재실행
        indexer_client.run_indexer(INDEXER_NAME)
        return "running"
    except HttpResponseError as e:
        print(f"Azure Search error: {str(e)}")
        return f"error: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"error: {str(e)}"

def get_indexer_status():
    try:
        AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
        AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
        INDEXER_NAME = os.environ.get("INDEXER_NAME", "azureblob-indexer")
        
        # Verify AZURE_SEARCH_KEY is a string
        if not isinstance(AZURE_SEARCH_KEY, str):
            logger.error(f"AZURE_SEARCH_KEY is not a string: {type(AZURE_SEARCH_KEY)}")
            return "error: AZURE_SEARCH_KEY must be a string"
            
        # Verify INDEXER_NAME is a string
        if not isinstance(INDEXER_NAME, str):
            logger.error(f"INDEXER_NAME is not a string: {type(INDEXER_NAME)}")
            return "error: INDEXER_NAME must be a string"
            
        indexer_client = SearchIndexerClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        try:
            status = indexer_client.get_indexer_status(INDEXER_NAME)
            return status.last_result.status
        except ResourceNotFoundError:
            # 인덱서가 없으면 생성 필요
            return "not_found"
    except HttpResponseError as e:
        logger.error(f"Azure Search error: {str(e)}")
        return f"error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in get_indexer_status: {str(e)}")
        return f"error: {str(e)}"
