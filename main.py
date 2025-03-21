import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexerClient, SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndexer,
    SearchIndex,
    InputFieldMappingEntry,
    SearchIndexerDataSourceConnection,
    SearchFieldDataType,
    SimpleField,
    SearchableField
)
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
import openai
from openai import AzureOpenAI
import time
from azure.search.documents import SearchClient
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv package not found. Using default environment variables.")

# Define standardized response models
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str

# Create a single FastAPI app instance
app = FastAPI(
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure configuration from environment variables (without default values exposing credentials)
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "azureblob-index")
DATASOURCE_NAME = os.environ.get("DATASOURCE_NAME", "simplerag")
INDEXER_NAME = os.environ.get("INDEXER_NAME", "azureblob-indexer")

# Azure OpenAI configuration
AZURE_OPENAI_API_ENDPOINT = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME")

# Check for required environment variables at startup
required_env_vars = {
    "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION_STRING,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
    "AZURE_OPENAI_API_ENDPOINT": AZURE_OPENAI_API_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_MODEL_NAME": AZURE_OPENAI_MODEL_NAME,
    "CONTAINER_NAME": CONTAINER_NAME
}

missing_vars = [var for var, value in required_env_vars.items() if not value]
if missing_vars:
    missing = ', '.join(missing_vars)
    logger.error(f"Missing required environment variables: {missing}")
    raise RuntimeError(f"Missing required environment variables: {missing}")

# Configure OpenAI client (only if credentials are available)
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_API_ENDPOINT:
    openai.api_type = "azure"
    openai.api_base = AZURE_OPENAI_API_ENDPOINT
    openai.api_version = AZURE_OPENAI_API_VERSION
    openai.api_key = AZURE_OPENAI_API_KEY

# Blob의 기존 PDF 삭제 함수
def delete_existing_pdfs():
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        blobs = container_client.list_blobs()
        for blob in blobs:
            container_client.delete_blob(blob.name)
        return True
    except Exception as e:
        logger.error(f"Error deleting PDFs: {str(e)}")
        return False

# Azure Search 리소스 생성 함수
def create_search_resources():
    try:
        import requests
        
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

# Azure Search Indexer와 Index를 재설정하는 함수
def reset_search_resources():
    try:
        # 인덱서 클라이언트 생성
        indexer_client = SearchIndexerClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        # 인덱스 클라이언트 생성
        index_client = SearchIndexClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        # 1. 인덱서 리셋
        try:
            indexer_client.reset_indexer(INDEXER_NAME)
            logger.info(f"Indexer '{INDEXER_NAME}' reset successfully")
        except ResourceNotFoundError:
            logger.info(f"Indexer '{INDEXER_NAME}' not found, will be created")
        except Exception as e:
            logger.error(f"Error resetting indexer: {str(e)}")
        
        # 2. 인덱스 내용 비우기 (인덱스를 삭제하고 다시 생성)
        try:
            # 기존 인덱스 삭제 시도
            index_client.delete_index(INDEX_NAME)
            logger.info(f"Index '{INDEX_NAME}' deleted successfully")
        except ResourceNotFoundError:
            logger.info(f"Index '{INDEX_NAME}' not found, will be created")
        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
        
        # 필요한 리소스 재생성
        return create_search_resources()
        
    except Exception as e:
        logger.error(f"Error resetting search resources: {str(e)}")
        return False

# Azure Search 리소스를 완전히 삭제하고 다시 생성하는 함수
def delete_and_recreate_search_resources():
    try:
        import requests
        
        # Common headers for all requests
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_KEY
        }
        
        # 1. Delete existing indexer if it exists
        indexer_url = f"{AZURE_SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(indexer_url, headers=headers)
            if delete_response.status_code == 204 or delete_response.status_code == 404:
                logger.info(f"Indexer '{INDEXER_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Failed to delete indexer: {delete_response.text}")
        except Exception as e:
            logger.warning(f"Error deleting indexer: {str(e)}")
        
        # 2. Delete existing index if it exists
        index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(index_url, headers=headers)
            if delete_response.status_code == 204 or delete_response.status_code == 404:
                logger.info(f"Index '{INDEX_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Failed to delete index: {delete_response.text}")
        except Exception as e:
            logger.warning(f"Error deleting index: {str(e)}")
        
        # 3. Delete existing data source if it exists
        datasource_url = f"{AZURE_SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(datasource_url, headers=headers)
            if delete_response.status_code == 204 or delete_response.status_code == 404:
                logger.info(f"Data source '{DATASOURCE_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Failed to delete data source: {delete_response.text}")
        except Exception as e:
            logger.warning(f"Error deleting data source: {str(e)}")
        
        # Add a small delay to ensure deletion is complete
        time.sleep(5)
        
        # 4. Create new resources
        return create_search_resources()
        
    except Exception as e:
        logger.error(f"Error recreating search resources: {str(e)}")
        return False

# Azure Search Indexer와 Index를 확실히 재설정하고 재생성하는 함수
def reset_and_run_indexer():
    try:
        logger.info("Starting to reset and recreate both index and indexer via direct API calls...")
        import requests
        
        # Common headers for all requests
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_KEY
        }
        
        # 1. Delete the indexer first
        indexer_url = f"{AZURE_SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(indexer_url, headers=headers)
            if delete_response.status_code in [204, 404]:
                logger.info(f"Indexer '{INDEXER_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Error deleting indexer: {delete_response.status_code} - {delete_response.text}")
        except Exception as e:
            logger.warning(f"Exception when deleting indexer: {str(e)}")
        
        # 2. Delete the index - IMPORTANT for resetting all data
        index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(index_url, headers=headers)
            if delete_response.status_code in [204, 404]:
                logger.info(f"Index '{INDEX_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Error deleting index: {delete_response.status_code} - {delete_response.text}")
        except Exception as e:
            logger.warning(f"Exception when deleting index: {str(e)}")
        
        # 3. Make sure data source is also reset
        datasource_url = f"{AZURE_SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version=2020-06-30"
        try:
            delete_response = requests.delete(datasource_url, headers=headers)
            if delete_response.status_code in [204, 404]:
                logger.info(f"Data source '{DATASOURCE_NAME}' deleted or did not exist")
            else:
                logger.warning(f"Error deleting data source: {delete_response.status_code} - {delete_response.text}")
        except Exception as e:
            logger.warning(f"Exception when deleting data source: {str(e)}")
        
        # Wait to ensure Azure has completed the deletion operations
        logger.info("Waiting for deletion operations to complete...")
        time.sleep(5)
        
        # 4. Recreate everything from scratch
        logger.info("Creating new search resources...")
        recreate_result = create_search_resources()
        
        if not recreate_result:
            logger.error("Failed to recreate search resources")
            return "error: Failed to recreate search resources"
        
        logger.info("Successfully reset and recreated search resources (index and indexer)")
        return "running"
    except Exception as e:
        logger.error(f"Unexpected error in reset_and_run_indexer: {str(e)}")
        return f"error: {str(e)}"

# 인덱싱 상태를 확인하는 함수
def get_indexer_status():
    try:
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

# 인덱스가 존재하는지 확인하는 함수
def check_index_exists():
    try:
        import requests
        
        # Index URL
        index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
        
        # Headers
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_KEY
        }
        
        # Send request
        response = requests.get(index_url, headers=headers)
        
        # Return true if index exists (status code 200)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking if index exists: {str(e)}")
        return False

# Azure Search 리소스를 강제로 삭제하는 함수 (더 확실한 방법으로)
def force_delete_search_resources():
    try:
        import requests
        
        # Common headers for all requests
        headers = {
            'Content-Type': 'application/json',
            'api-key': AZURE_SEARCH_KEY
        }
        
        # Log the start of process
        logger.info("Starting forced deletion of all search resources")
        
        # 1. First check and delete the indexer
        indexer_url = f"{AZURE_SEARCH_ENDPOINT}/indexers/{INDEXER_NAME}?api-version=2020-06-30"
        try:
            # Check if indexer exists
            check_response = requests.get(indexer_url, headers=headers)
            if check_response.status_code == 200:
                # Indexer exists, delete it
                delete_response = requests.delete(indexer_url, headers=headers)
                if delete_response.status_code in [204, 404]:
                    logger.info(f"Indexer '{INDEXER_NAME}' successfully deleted")
                else:
                    logger.warning(f"Failed to delete indexer: {delete_response.status_code} - {delete_response.text}")
            else:
                logger.info(f"Indexer '{INDEXER_NAME}' does not exist, no need to delete")
        except Exception as e:
            logger.warning(f"Error during indexer deletion check: {str(e)}")
        
        # 2. Wait to ensure indexer is deleted before deleting index
        time.sleep(3)
        
        # 3. Now check and delete the index
        index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
        try:
            # Check if index exists
            check_response = requests.get(index_url, headers=headers)
            if check_response.status_code == 200:
                # Index exists, delete it
                delete_response = requests.delete(index_url, headers=headers)
                if delete_response.status_code in [204, 404]:
                    logger.info(f"Index '{INDEX_NAME}' successfully deleted")
                else:
                    logger.warning(f"Failed to delete index: {delete_response.status_code} - {delete_response.text}")
            else:
                logger.info(f"Index '{INDEX_NAME}' does not exist, no need to delete")
        except Exception as e:
            logger.warning(f"Error during index deletion check: {str(e)}")
        
        # 4. Wait to ensure index is deleted
        time.sleep(3)
        
        # 5. Finally check and delete the data source
        datasource_url = f"{AZURE_SEARCH_ENDPOINT}/datasources/{DATASOURCE_NAME}?api-version=2020-06-30"
        try:
            # Check if data source exists
            check_response = requests.get(datasource_url, headers=headers)
            if check_response.status_code == 200:
                # Data source exists, delete it
                delete_response = requests.delete(datasource_url, headers=headers)
                if delete_response.status_code in [204, 404]:
                    logger.info(f"Data source '{DATASOURCE_NAME}' successfully deleted")
                else:
                    logger.warning(f"Failed to delete data source: {delete_response.status_code} - {delete_response.text}")
            else:
                logger.info(f"Data source '{DATASOURCE_NAME}' does not exist, no need to delete")
        except Exception as e:
            logger.warning(f"Error during data source deletion check: {str(e)}")
        
        # 6. Final check to ensure all resources are really gone
        all_deleted = True
        
        # Check indexer
        check_response = requests.get(indexer_url, headers=headers)
        if check_response.status_code == 200:
            logger.warning(f"Indexer '{INDEXER_NAME}' still exists after deletion attempt")
            all_deleted = False
        
        # Check index
        check_response = requests.get(index_url, headers=headers)
        if check_response.status_code == 200:
            logger.warning(f"Index '{INDEX_NAME}' still exists after deletion attempt")
            all_deleted = False
        
        # Check data source
        check_response = requests.get(datasource_url, headers=headers)
        if check_response.status_code == 200:
            logger.warning(f"Data source '{DATASOURCE_NAME}' still exists after deletion attempt")
            all_deleted = False
        
        # Wait a bit more to ensure Azure processes the deletion
        time.sleep(5)
        
        # Return result
        return all_deleted
        
    except Exception as e:
        logger.error(f"Error in force_delete_search_resources: {str(e)}")
        return False

@app.post("/reset-search")
async def reset_search_endpoint():
    """Endpoint to completely delete and recreate the search index and indexer."""
    try:
        logger.info("API request received to reset search resources (index and indexer)")
        
        # Check if required credentials are available
        if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
            return StandardResponse(
                success=False,
                message="Missing Azure Search credentials. Please configure environment variables.",
                timestamp=datetime.now().isoformat()
            )
        
        # First ensure all resources are deleted using the improved method
        deletion_success = force_delete_search_resources()
        
        if not deletion_success:
            logger.warning("Could not confirm complete deletion of search resources")
        
        # Add additional delay to ensure Azure backend has processed the deletions
        time.sleep(5)
        
        # Now create resources from scratch
        creation_result = create_search_resources()
        
        if not creation_result:
            return StandardResponse(
                success=False,
                message="Search resources were deleted but could not be recreated",
                data={"status": "error"},
                timestamp=datetime.now().isoformat()
            )
        
        return StandardResponse(
            success=True,
            message="인덱스 및 인덱서가 성공적으로 재설정되었습니다.",
            data={"status": "success"},
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in reset_search_endpoint: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Failed to reset search resources: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Validate required environment variables
        if not AZURE_STORAGE_CONNECTION_STRING:
            return StandardResponse(
                success=False,
                message="Missing Azure Storage connection string. Please configure environment variables.",
                timestamp=datetime.now().isoformat()
            )

        # 기존 PDF 삭제
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)

        # 컨테이너 존재 확인 및 생성
        try:
            container_client.get_container_properties()
        except:
            blob_service.create_container(CONTAINER_NAME)
            container_client = blob_service.get_container_client(CONTAINER_NAME)

        # Delete existing blobs
        blobs = container_client.list_blobs()
        for blob in blobs:
            container_client.delete_blob(blob.name)

        # 새 PDF 업로드
        blob_client = blob_service.get_blob_client(container=CONTAINER_NAME, blob=file.filename)
        content = await file.read()
        blob_client.upload_blob(content, overwrite=True)

        # Use the improved force delete function to ensure all resources are deleted
        try:
            logger.info("Forcefully deleting all search resources before recreation")
            force_delete_search_resources()
            
            # Add a delay to ensure deletion is processed
            time.sleep(5)
            
            # Verify the index was deleted
            if check_index_exists():
                logger.warning("Index still exists after deletion attempt - trying delete again")
                # Try direct delete one more time
                import requests
                headers = {
                    'Content-Type': 'application/json',
                    'api-key': AZURE_SEARCH_KEY
                }
                index_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2020-06-30"
                requests.delete(index_url, headers=headers)
                time.sleep(3)
            
            # Now create new resources
            logger.info("Creating new search resources after deletion")
            recreate_result = create_search_resources()
            
            if not recreate_result:
                return StandardResponse(
                    success=False,
                    message="Failed to recreate search resources",
                    timestamp=datetime.now().isoformat()
                )
            
            # Set indexer status
            indexer_status = "created_and_running"
                
        except Exception as e:
            logger.error(f"Error during search resource recreation: {str(e)}")
            indexer_status = f"error: {str(e)}"
            
        return StandardResponse(
            success=True,
            message="파일 업로드 완료 및 검색 리소스 재생성",
            data={"filename": file.filename, "indexer_status": indexer_status},
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Upload failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.get("/indexer-status")
async def indexer_status():
    try:
        # Check that required environment variables are set and are strings
        if not AZURE_SEARCH_ENDPOINT or not isinstance(AZURE_SEARCH_ENDPOINT, str):
            return StandardResponse(
                success=False,
                message="AZURE_SEARCH_ENDPOINT not properly configured",
                data={"indexing_status": "error: configuration_missing"},
                timestamp=datetime.now().isoformat()
            )

        if not AZURE_SEARCH_KEY or not isinstance(AZURE_SEARCH_KEY, str):
            return StandardResponse(
                success=False,
                message="AZURE_SEARCH_KEY not properly configured",
                data={"indexing_status": "error: configuration_missing"},
                timestamp=datetime.now().isoformat()
            )

        status = get_indexer_status()

        # If the indexer status is "reset", trigger the indexer to run immediately.
        if status == "reset":
            try:
                indexer_client = SearchIndexerClient(
                    endpoint=AZURE_SEARCH_ENDPOINT,
                    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
                )
                indexer_client.run_indexer(INDEXER_NAME)
                status = "running"
                logger.info("Indexer was reset; now executing indexer_run to resume indexing.")
            except Exception as e:
                logger.error(f"Error running indexer after reset: {str(e)}")
                return StandardResponse(
                    success=False,
                    message=f"Failed to run indexer after reset: {str(e)}",
                    data={"indexing_status": status},
                    timestamp=datetime.now().isoformat()
                )

        message = "인덱싱 상태 확인 완료"
        if status == "not_found":
            message = "인덱서가 존재하지 않습니다. PDF를 업로드하여 인덱서를 생성하세요."

        return StandardResponse(
            success=True,
            message=message,
            data={"indexing_status": status},
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in indexer_status endpoint: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Failed to get indexer status: {str(e)}",
            timestamp=datetime.now().isoformat()
        )
@app.get("/upload-status")
async def check_any_files():
    """Check if any file is already in the container."""
    try:
        blob_service = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service.get_container_client(CONTAINER_NAME)
        
        # Verify container exists
        try:
            container_client.get_container_properties()
        except:
            return StandardResponse(
                success=False,
                message="Container not found. Please upload a file first.",
                timestamp=datetime.now().isoformat()
            )
        
        # List all blobs
        blobs = list(container_client.list_blobs())
        if blobs:
            filenames = [blob.name for blob in blobs]
            return StandardResponse(
                success=True,
                message="Files found.",
                data={"filenames": filenames},
                timestamp=datetime.now().isoformat()
            )
        else:
            return StandardResponse(
                success=False,
                message="No files found in the container.",
                timestamp=datetime.now().isoformat()
            )
    except Exception as e:
        logger.error(f"Check file status error: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Error checking file status: {str(e)}",
            timestamp=datetime.now().isoformat()
        )
def search_pdf_content(query: str, top_k=3) -> dict:
    """Query Azure Search index for relevant PDF content and return structured results."""
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        results = search_client.search(
            search_text=query,
            select=["content", "metadata_storage_name", "metadata_storage_path"],
            top=top_k
        )
        
        documents = []
        for result in results:
            filename = result.get("metadata_storage_name", "Unknown document")
            content = result.get("content", "")
            path = result.get("metadata_storage_path", "")
            if content:
                # Limit content length to avoid overly long prompts
                content_snippet = content[:2000] + ("..." if len(content) > 2000 else "")
                documents.append({
                    "filename": filename,
                    "content": content_snippet,
                    "path": path
                })
        
        return {
            "documents": documents,
            "count": len(documents),
            "query": query
        }
            
    except Exception as e:
        print(f"Search error: {str(e)}")
        return {"error": str(e), "documents": [], "count": 0, "query": query}

async def get_pdf_content(query="*") -> dict:
    """Retrieve all uploaded PDF content from the Azure Search index."""
    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY)
        )
        
        # Use "*" as a wildcard query to get all documents
        results = search_client.search(
            search_text=query,
            select=["content", "metadata_storage_name", "metadata_storage_path"],
            top=100  # Adjust based on expected document count
        )
        
        documents = []
        for result in results:
            filename = result.get("metadata_storage_name", "Unknown document")
            content = result.get("content", "")
            path = result.get("metadata_storage_path", "")
            if content:
                documents.append({
                    "filename": filename,
                    "content": content,
                    "path": path
                })
        
        return {
            "documents": documents,
            "count": len(documents)
        }
            
    except Exception as e:
        print(f"Error retrieving PDF content: {str(e)}")
        return {"error": str(e), "documents": [], "count": 0}

@app.get("/pdf-content")
async def get_pdf_content_endpoint(query: str):
    """Endpoint to display raw PDF content retrieved from Azure Search."""
    try:
        status = get_indexer_status()
        
        if status == "not_found":
            return StandardResponse(
                success=False,
                message="먼저 PDF를 업로드하여 인덱서를 생성하세요.",
                data={"status": "not_found"},
                timestamp=datetime.now().isoformat()
            )
        
        if isinstance(status, str) and status.startswith("error"):
            return StandardResponse(
                success=False,
                message=f"인덱서 상태 확인 중 오류 발생: {status}",
                data={"status": "error"},
                timestamp=datetime.now().isoformat()
            )
        
        if status != "success":
            return StandardResponse(
                success=False,
                message="파일이 아직 인덱싱 중입니다. 잠시 후 다시 시도하세요.",
                data={"status": status},
                timestamp=datetime.now().isoformat()
            )

        search_results = search_pdf_content(query)
        return StandardResponse(
            success=True,
            message="PDF 콘텐츠 검색 완료",
            data=search_results,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            message=f"Failed to retrieve PDF content: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.post("/chat")
async def chat(prompt: str):
    try:
        # Validate required environment variables
        if not all([AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_API_ENDPOINT, AZURE_OPENAI_API_KEY]):
            return StandardResponse(
                success=False,
                message="Missing required Azure credentials. Please configure environment variables.",
                timestamp=datetime.now().isoformat()
            )

        # 인덱싱 상태 확인
        status = get_indexer_status()
        
        if status == "not_found":
            return StandardResponse(
                success=False,
                message="먼저 PDF를 업로드하여 인덱서를 생성하세요.",
                data={"status": "not_found"},
                timestamp=datetime.now().isoformat()
            )
        
        if isinstance(status, str) and status.startswith("error"):
            return StandardResponse(
                success=False,
                message=f"인덱서 상태 확인 중 오류 발생: {status}",
                data={"status": "error"},
                timestamp=datetime.now().isoformat()
            )
        
        if status != "success":
            return StandardResponse(
                success=False,
                message="파일이 아직 인덱싱 중입니다. 잠시 후 다시 시도하세요.",
                data={"status": status},
                timestamp=datetime.now().isoformat()
            )

        # Just retrieve all PDF content without searching - properly await the coroutine
        pdf_results = await get_pdf_content("*")
        
        # Format the content for inclusion in the prompt
        pdf_context = ""
        sources = []
        
        if pdf_results["count"] > 0:
            for i, doc in enumerate(pdf_results["documents"]):
                pdf_context += f"Document {i+1}: {doc['filename']}\n"
                pdf_context += f"Content: {doc['content']}\n\n"
                sources.append({"filename": doc['filename'], "path": doc['path']})
        else:
            pdf_context = "No uploaded PDF content found."

        # Create system message with the PDF content
        system_message = """You are an AI assistant that helps answer questions based on PDF documents.
Answer based ONLY on the content in the documents provided below.
If the information isn't in the documents, clearly state that.

Here is the content from the uploaded PDF documents:

"""
        system_message += pdf_context

        # Fix the OpenAI client initialization
        try:
            # Create the OpenAI client directly using the imported class
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_API_ENDPOINT
                # Removed any 'proxies' argument here
            )
           
            
            response =  client.chat.completions.create(
                model=AZURE_OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                top_p=0.95
            )
            
            answer = response.choices[0].message.content
            
            # Return the answer and sources
            return StandardResponse(
                success=True,
                message="챗봇 응답 생성 완료",
                data={
                    "answer": answer, 
                    "sources": sources
                },
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return StandardResponse(
                success=False,
                message=f"OpenAI API error: {str(e)}",
                timestamp=datetime.now().isoformat()
            )
            
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Chat failed: {str(e)}",
            timestamp=datetime.now().isoformat()
        )

@app.get("/health")
async def health():
    """Lightweight health check endpoint for Azure's HTTP probes."""
    return {"status": "healthy", "service": "simplerag"}

# if __name__ == "__main__":
#     # Get port from environment variable or use default
#     port = int(os.environ.get("PORT", 8000))
#     host = os.environ.get("HOST", "0.0.0.0")
    
#     logger.info(f"Starting FastAPI server on {host}:{port}...")
#     logger.info(f"Visit http://{host}:{port}/docs for API documentation")
  
#     uvicorn.run(app, host=host, port=port)