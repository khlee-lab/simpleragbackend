import os
import time
import asyncio
import logging
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexerClient, SearchIndexClient
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient
import openai
from openai import AzureOpenAI
from pydantic import BaseModel

# =========================================================
# Setup Logging
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# Try to load dotenv
# =========================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv package not found. Using default environment variables.")

# =========================================================
# Environment Variables
# =========================================================
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME", "azureblob-index")
DATASOURCE_NAME = os.environ.get("DATASOURCE_NAME", "simplerag")
INDEXER_NAME = os.environ.get("INDEXER_NAME", "azureblob-indexer")
AZURE_SEARCH_API_VERSION = "2024-07-01"

# Azure OpenAI
AZURE_OPENAI_API_ENDPOINT = os.environ.get("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_MODEL_NAME = os.environ.get("AZURE_OPENAI_MODEL_NAME")

# =========================================================
# Validate required environment variables
# =========================================================
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
    msg = "Missing required environment variables: " + ", ".join(missing_vars)
    logger.error(msg)
    raise RuntimeError(msg)

# =========================================================
# Configure OpenAI
# =========================================================
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_API_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY

# =========================================================
# Pydantic Response Model
# =========================================================
class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str

# =========================================================
# Helper Classes
# =========================================================
class AzureBlobManager:
    """Azure Blob Storage の操作をまとめたクラス。"""
    
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.ensure_container_exists()

    def ensure_container_exists(self):
        """コンテナが存在しない場合は作成。"""
        try:
            self.container_client.get_container_properties()
        except:
            logger.info(f"Container '{self.container_name}' does not exist. Creating now...")
            self.blob_service_client.create_container(self.container_name)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def delete_all_blobs(self):
        """コンテナ内の全てのBlobを削除。"""
        try:
            blobs = self.container_client.list_blobs()
            for blob in blobs:
                self.container_client.delete_blob(blob.name)
                logger.info(f"Deleted blob: {blob.name}")
        except Exception as e:
            logger.error(f"Error deleting blobs: {str(e)}")
            raise

    def upload_blob(self, file_name: str, content: bytes):
        """Blobをアップロードする。"""
        try:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file_name)
            blob_client.upload_blob(content, overwrite=True)
            logger.info(f"Uploaded new file: {file_name}")
        except Exception as e:
            logger.error(f"Error uploading blob: {str(e)}")
            raise


class AzureSearchManager:
    """Azure Cognitive Search の操作をまとめたクラス。"""

    def __init__(self, endpoint: str, api_key: str, api_version: str, 
                 index_name: str, datasource_name: str, indexer_name: str,
                 storage_connection_string: str, container_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.index_name = index_name
        self.datasource_name = datasource_name
        self.indexer_name = indexer_name
        self.storage_connection_string = storage_connection_string
        self.container_name = container_name

        self.indexer_client = SearchIndexerClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )

    def create_search_resources(self) -> bool:
        """
        データソース・インデックス・インデクサーを作成（存在しない場合は作成、存在すれば更新）。
        失敗時は False を返す。
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }
            # 1. データソース作成
            data_source_url = f"{self.endpoint}/datasources/{self.datasource_name}?api-version={self.api_version}"
            data_source_body = {
                "name": self.datasource_name,
                "type": "azureblob",
                "credentials": {
                    "connectionString": self.storage_connection_string
                },
                "container": {
                    "name": self.container_name
                }
            }
            ds_resp = requests.put(data_source_url, headers=headers, json=data_source_body)
            ds_resp.raise_for_status()
            logger.info(f"Data source '{self.datasource_name}' created/updated.")

            # 2. インデックス作成
            index_url = f"{self.endpoint}/indexes/{self.index_name}?api-version={self.api_version}"
            index_body = {
                "name": self.index_name,
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
                        "retrievable": True,
                        "analyzer": "standard.lucene"
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
                ],
                "semantic": {
                    "configurations": [
                        {
                            "name": "default",
                            "prioritizedFields": {
                                "titleField": None,
                                "contentFields": [
                                    {
                                        "fieldName": "content"
                                    }
                                ],
                                "keywordsFields": []
                            }
                        }
                    ]
                }
            }
            idx_resp = requests.put(index_url, headers=headers, json=index_body)
            idx_resp.raise_for_status()
            logger.info(f"Index '{self.index_name}' created/updated.")

            # 3. インデクサー作成
            indexer_url = f"{self.endpoint}/indexers/{self.indexer_name}?api-version={self.api_version}"
            indexer_body = {
                "name": self.indexer_name,
                "dataSourceName": self.datasource_name,
                "targetIndexName": self.index_name,
                "parameters": {
                    "configuration": {
                        "parsingMode": "default",
                        "dataToExtract": "contentAndMetadata"
                    }
                },
                "fieldMappings": [
                    {
                        "sourceFieldName": "metadata_storage_path",
                        "targetFieldName": "metadata_storage_path",
                        "mappingFunction": {
                            "name": "base64Encode"
                        }
                    }
                ]
            }
            inx_resp = requests.put(indexer_url, headers=headers, json=indexer_body)
            inx_resp.raise_for_status()
            logger.info(f"Indexer '{self.indexer_name}' created/updated.")
            time.sleep(2)

            # 4. インデクサーの実行
            run_url = f"{self.endpoint}/indexers/{self.indexer_name}/run?api-version={self.api_version}"
            run_response = requests.post(run_url, headers=headers)
            run_response.raise_for_status()
            logger.info(f"Indexer '{self.indexer_name}' triggered to run.")

            return True

        except Exception as e:
            logger.error(f"Error creating search resources: {str(e)}")
            return False

    def delete_search_resources(self) -> Tuple[bool, List[str]]:
        """
        Azure Search リソース（インデクサー、インデックス、データソース）を削除する。
        戻り値: (成功フラグ, 発生したエラーリスト)
        """
        errors = []
        try:
            # インデクサー削除
            try:
                self.indexer_client.delete_indexer(self.indexer_name)
                logger.info(f"Indexer '{self.indexer_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Indexer '{self.indexer_name}' not found.")
            except Exception as e:
                msg = f"Failed to delete indexer: {str(e)}"
                errors.append(msg)
                logger.error(msg)

            # インデックス削除
            try:
                self.index_client.delete_index(self.index_name)
                logger.info(f"Index '{self.index_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Index '{self.index_name}' not found.")
            except Exception as e:
                msg = f"Failed to delete index: {str(e)}"
                errors.append(msg)
                logger.error(msg)

            # データソース削除
            try:
                self.indexer_client.delete_data_source_connection(self.datasource_name)
                logger.info(f"Data source '{self.datasource_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Data source '{self.datasource_name}' not found.")
            except Exception as e:
                msg = f"Failed to delete data source: {str(e)}"
                errors.append(msg)
                logger.error(msg)

        except Exception as e:
            msg = f"Unexpected error while deleting resources: {str(e)}"
            errors.append(msg)
            logger.error(msg)

        return (len(errors) == 0, errors)

    def force_delete_search_resources(self) -> bool:
        """検索リソースを強制的に削除し、実際に消えたかチェックする。"""
        success, errors = self.delete_search_resources()
        time.sleep(5)  # Azureバックエンドが削除処理を反映するのを待つ
        if not success:
            logger.warning(f"Some errors occurred while deleting: {errors}")

        # 確認
        all_deleted = self.confirm_resources_deleted()
        return all_deleted

    def confirm_resources_deleted(self) -> bool:
        """
        リソースが実際に削除されたか REST API で確認。
        すべて 404 なら True。
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }
            all_deleted = True

            # Check indexer
            indexer_url = f"{self.endpoint}/indexers/{self.indexer_name}?api-version={self.api_version}"
            res_idxr = requests.get(indexer_url, headers=headers)
            if res_idxr.status_code == 200:
                logger.warning(f"Indexer '{self.indexer_name}' still exists.")
                all_deleted = False

            # Check index
            index_url = f"{self.endpoint}/indexes/{self.index_name}?api-version={self.api_version}"
            res_index = requests.get(index_url, headers=headers)
            if res_index.status_code == 200:
                logger.warning(f"Index '{self.index_name}' still exists.")
                all_deleted = False

            # Check data source
            ds_url = f"{self.endpoint}/datasources/{self.datasource_name}?api-version={self.api_version}"
            res_ds = requests.get(ds_url, headers=headers)
            if res_ds.status_code == 200:
                logger.warning(f"Data source '{self.datasource_name}' still exists.")
                all_deleted = False

            return all_deleted

        except Exception as e:
            logger.error(f"Error in confirm_resources_deleted: {str(e)}")
            return False

    def reset_indexer(self) -> bool:
        """インデクサーの実行状態リセット。"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }
            reset_url = f"{self.endpoint}/indexers/{self.indexer_name}/reset?api-version={self.api_version}"
            response = requests.post(reset_url, headers=headers)
            if response.status_code in [204, 200]:
                logger.info(f"Indexer '{self.indexer_name}' reset.")
                return True
            else:
                logger.warning(f"Failed to reset indexer: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error resetting indexer: {str(e)}")
            return False

    def run_indexer(self) -> bool:
        """インデクサーを明示的に実行する。"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }
            run_url = f"{self.endpoint}/indexers/{self.indexer_name}/run?api-version={self.api_version}"
            response = requests.post(run_url, headers=headers)
            if response.status_code in [202, 204]:
                logger.info(f"Indexer '{self.indexer_name}' run triggered.")
                return True
            else:
                logger.warning(f"Failed to run indexer: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error running indexer: {str(e)}")
            return False

    def get_indexer_status(self) -> str:
        """
        インデクサーの最終実行結果ステータスを返す。
        戻り値が "not_found" の場合はインデクサー自体が存在しない。
        "error:" で始まる場合は異常。
        """
        try:
            # 最新バージョン指定でクライアントを作り直す
            indexer_client = SearchIndexerClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            status = indexer_client.get_indexer_status(self.indexer_name)
            return status.last_result.status  # "success", "inProgress" 等
        except ResourceNotFoundError:
            return "not_found"
        except HttpResponseError as e:
            logger.error(f"Azure Search error: {str(e)}")
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in get_indexer_status: {str(e)}")
            return f"error: {str(e)}"

    def search_pdf_content(self, query: str, top_k=3) -> dict:
        """
        Azure Search Index を検索し、該当する PDF コンテンツを返す。
        """
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            search_options = {
                "select": ["content", "metadata_storage_name", "metadata_storage_path"],
                "top": top_k,
                "query_type": "simple",
                "query_language": "ja-JP"
            }
            results = search_client.search(query, **search_options)

            documents = []
            for result in results:
                filename = result.get("metadata_storage_name", "Unknown document")
                content = result.get("content", "")
                path = result.get("metadata_storage_path", "")
                if content:
                    snippet = content[:2000] + ("..." if len(content) > 2000 else "")
                    documents.append({
                        "filename": filename,
                        "content": snippet,
                        "path": path
                    })
            return {"documents": documents, "count": len(documents), "query": query}

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"error": str(e), "documents": [], "count": 0, "query": query}

    async def get_all_pdf_content(self, query="*") -> dict:
        """
        インデックス内の全 PDF コンテンツを取得する（検索クエリを "*" とする）。
        """
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            search_options = {
                "select": ["content", "metadata_storage_name", "metadata_storage_path"],
                "top": 100
            }
            results = search_client.search(query, **search_options)

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

            return {"documents": documents, "count": len(documents)}
        except Exception as e:
            logger.error(f"Error retrieving PDF content: {str(e)}")
            return {"error": str(e), "documents": [], "count": 0}


class AzureOpenAIManager:
    """Azure OpenAI の操作をまとめたクラス。"""

    def __init__(self, endpoint: str, api_key: str, api_version: str, model_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model_name = model_name

        # OpenAIライブラリ設定はグローバルに適用されるため、
        # ここではあえて設定を参照するだけにしている。

    def chat_completion(self, system_message: str, user_prompt: str, temperature=0.7, top_p=0.95) -> str:
        """チャット形式のCompletionを取得。"""
        try:
            client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise


# =========================================================
# FastAPI App
# =========================================================
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Instantiate Managers
# =========================================================
blob_manager = AzureBlobManager(
    connection_string=AZURE_STORAGE_CONNECTION_STRING,
    container_name=CONTAINER_NAME
)
search_manager = AzureSearchManager(
    endpoint=AZURE_SEARCH_ENDPOINT,
    api_key=AZURE_SEARCH_KEY,
    api_version=AZURE_SEARCH_API_VERSION,
    index_name=INDEX_NAME,
    datasource_name=DATASOURCE_NAME,
    indexer_name=INDEXER_NAME,
    storage_connection_string=AZURE_STORAGE_CONNECTION_STRING,
    container_name=CONTAINER_NAME
)
openai_manager = AzureOpenAIManager(
    endpoint=AZURE_OPENAI_API_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    model_name=AZURE_OPENAI_MODEL_NAME
)

# =========================================================
# Endpoints
# =========================================================
@app.post("/reset-search", response_model=StandardResponse, summary="Reset Search Index and Indexer", tags=["Search Management"])
async def reset_search_endpoint():
    """
    インデックスとインデクサーを完全に削除して再作成するエンドポイント
    """
    try:
        logger.info("API request received to reset search resources (index and indexer).")

        # 強制削除
        deletion_success = search_manager.force_delete_search_resources()
        if not deletion_success:
            logger.warning("Could not confirm complete deletion of search resources.")

        await asyncio.sleep(5)  # バックエンドでの削除反映待ち

        # 再作成
        creation_result = search_manager.create_search_resources()
        if not creation_result:
            return StandardResponse(
                success=False,
                message="Search resources were deleted but could not be recreated.",
                data={"status": "error"},
                timestamp=datetime.now().isoformat()
            )

        return StandardResponse(
            success=True,
            message="インデックス及びインデクサーが再設定されました。",
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


@app.post("/upload", response_model=StandardResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF ファイルをアップロードし、コンテナ内の既存 PDF を削除してから、
    Azure Search リソースを再構築＆インデクサーを実行する。
    """
    try:
        # 1. 既存Blobの削除
        blob_manager.delete_all_blobs()

        # 2. 新たなPDFをアップロード
        content = await file.read()
        blob_manager.upload_blob(file.filename, content)

        # 3. 検索リソースを再作成
        search_manager.reset_indexer()  # まずインデクサーリセットしてから
        await asyncio.sleep(2)

        logger.info("Forcefully deleting all search resources before recreation.")
        success, errors = search_manager.delete_search_resources()
        if not success:
            logger.warning(f"Errors during resource deletion: {errors}")

        # Azure Search バックエンド反映待ち
        logger.info("Waiting for Azure Search to process deletion...")
        await asyncio.sleep(10)

        logger.info("Creating new search resources.")
        recreate_result = search_manager.create_search_resources()
        if not recreate_result:
            return StandardResponse(
                success=False,
                message="Failed to recreate search resources.",
                timestamp=datetime.now().isoformat()
            )

        # 4. インデクサ実行
        logger.info("Explicitly triggering indexer run.")
        run_result = search_manager.run_indexer()
        indexer_status = "created_and_running" if run_result else "created_but_not_running"

        return StandardResponse(
            success=True,
            message="ファイル アップロード完了、検索リソースを再作成しました。",
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


@app.post("/upload", response_model=StandardResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF ファイルをアップロードしたら、
    コンテナ内の既存 PDF を削除 → Azure Searchリソース(インデックス等)を削除 → 再作成 → インデクサ実行
    """
    try:
        # === 1. 既存Blobの削除 ===========================================
        blob_manager.delete_all_blobs()  # すべてのBlobを削除

        # === 2. 新しいPDFをアップロード ==================================
        content = await file.read()
        blob_manager.upload_blob(file.filename, content)

        # === 3. Azure Searchリソースを削除 ===============================
        logger.info("Deleting existing Azure Search resources...")
        success, errors = search_manager.delete_search_resources()
        if not success:
            logger.warning(f"Errors during resource deletion: {errors}")
        
        # Azureバックエンドの削除反映待ち (必要に応じて秒数を調整)
        await asyncio.sleep(5)

        # === 4. Azure Searchリソースを再作成 =============================
        logger.info("Creating new Azure Search resources...")
        recreate_result = search_manager.create_search_resources()
        if not recreate_result:
            return StandardResponse(
                success=False,
                message="Failed to recreate search resources.",
                timestamp=datetime.now().isoformat()
            )
        
        # === 5. インデクサーを明示的に実行 ================================
        logger.info("Explicitly triggering indexer run.")
        run_result = search_manager.run_indexer()
        indexer_status = "created_and_running" if run_result else "created_but_not_running"

        return StandardResponse(
            success=True,
            message="ファイル アップロード完了。検索リソースを再作成しました。",
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


@app.get("/upload-status", response_model=StandardResponse)
async def check_any_files():
    """
    コンテナ内にファイルがあるかどうかを確認する。
    """
    try:
        # コンテナが既に作成されていなければエラー
        blob_manager.ensure_container_exists()
        blobs = list(blob_manager.container_client.list_blobs())

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


@app.get("/pdf-content", response_model=StandardResponse)
async def get_pdf_content_endpoint(query: str):
    """
    指定クエリにマッチするPDFコンテンツを検索し、その内容を返す。
    インデクサーが存在するか / 動作が完了しているかを確認してから実行。
    """
    try:
        status = search_manager.get_indexer_status()

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

        search_results = search_manager.search_pdf_content(query)
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


@app.post("/chat", response_model=StandardResponse)
async def chat(prompt: str):
    """
    アップロードされたPDFの内容をもとにチャット形式で回答を生成するエンドポイント。
    """
    try:
        # インデクサーステータス確認
        status = search_manager.get_indexer_status()
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

        # すべてのPDFの内容を取得してプロンプトに含める
        pdf_results = await search_manager.get_all_pdf_content("*")

        pdf_context = ""
        sources = []
        if pdf_results["count"] > 0:
            for i, doc in enumerate(pdf_results["documents"]):
                pdf_context += f"Document {i+1}: {doc['filename']}\n"
                pdf_context += f"Content: {doc['content']}\n\n"
                sources.append({"filename": doc['filename'], "path": doc['path']})
        else:
            pdf_context = "No uploaded PDF content found."

        # システムメッセージ
        system_message = (
            "You are an AI assistant that helps answer questions based on PDF documents.\n"
            "Answer based ONLY on the content in the documents provided below.\n"
            "If the information isn't in the documents, clearly state that.\n\n"
            "Here is the content from the uploaded PDF documents:\n\n"
            f"{pdf_context}"
        )

        # OpenAI ChatCompletion
        answer = openai_manager.chat_completion(system_message, prompt)

        return StandardResponse(
            success=True,
            message="챗봇 응답 생성 완료",
            data={"answer": answer, "sources": sources},
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
    """
    ヘルスチェック用エンドポイント
    """
    return {"status": "healthy", "service": "simplerag"}


# # ローカル開発用(本番では別のプロセスマネージャ等で起動する想定)
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     host = os.environ.get("HOST", "0.0.0.0")
#     logger.info(f"Starting FastAPI server on {host}:{port}...")
#     uvicorn.run(app, host=host, port=port)
