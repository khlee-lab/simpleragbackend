import os
import time
import asyncio
import logging
import requests
from datetime import datetime
from typing import Any, Optional, List, Tuple, Dict

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Azure SDK
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexerClient, SearchIndexClient
from azure.search.documents import SearchClient
from azure.storage.blob import BlobServiceClient

# OpenAI SDK
import openai
from openai import AzureOpenAI

# Pydantic
from pydantic import BaseModel

# =========================================================
# ログ設定
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# .env読み込み（必要に応じてコメントアウト）
# =========================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv is not installed. Skipping .env loading.")

# =========================================================
# 環境変数の取得
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

# 必須環境変数の検証
required_env_vars = {
    "AZURE_STORAGE_CONNECTION_STRING": AZURE_STORAGE_CONNECTION_STRING,
    "CONTAINER_NAME": CONTAINER_NAME,
    "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
    "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
    "AZURE_OPENAI_API_ENDPOINT": AZURE_OPENAI_API_ENDPOINT,
    "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
    "AZURE_OPENAI_MODEL_NAME": AZURE_OPENAI_MODEL_NAME,
}

missing = [k for k, v in required_env_vars.items() if not v]
if missing:
    msg = f"Missing environment variables: {', '.join(missing)}"
    logger.error(msg)
    raise RuntimeError(msg)

# =========================================================
# OpenAIの設定
# =========================================================
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_API_ENDPOINT
openai.api_version = AZURE_OPENAI_API_VERSION
openai.api_key = AZURE_OPENAI_API_KEY

# =========================================================
# Pydanticモデル
# =========================================================
class StandardResponse(BaseModel):
    """APIの標準レスポンス用モデル。"""
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: str

# =========================================================
# Azure Blob管理クラス
# =========================================================
class AzureBlobManager:
    """Azure Blob Storage の操作をまとめたクラス。"""

    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.ensure_container_exists()

    def ensure_container_exists(self) -> None:
        """コンテナが存在しない場合は作成。"""
        try:
            self.container_client.get_container_properties()
        except Exception:
            logger.info(f"Container '{self.container_name}' does not exist. Creating...")
            self.blob_service_client.create_container(self.container_name)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)

    def delete_all_blobs(self) -> None:
        """コンテナ内のBlobをすべて削除する。"""
        try:
            for blob in self.container_client.list_blobs():
                self.container_client.delete_blob(blob.name)
                logger.info(f"Deleted blob: {blob.name}")
        except Exception as e:
            logger.error(f"Error deleting blobs: {str(e)}")
            raise

    def upload_blob(self, file_name: str, content: bytes) -> None:
        """Blobをアップロードする。

        Args:
            file_name (str): アップロード先のBlob名
            content (bytes): ファイルのバイナリ内容
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=file_name
            )
            blob_client.upload_blob(content, overwrite=True)
            logger.info(f"Uploaded file: {file_name}")
        except Exception as e:
            logger.error(f"Error uploading blob: {str(e)}")
            raise

# =========================================================
# Azure Search管理クラス
# =========================================================
class AzureSearchManager:
    """Azure Cognitive Search の操作をまとめたクラス。"""

    SUCCESSFUL_RUN_STATUS_CODES = {200, 202, 204}

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        index_name: str,
        datasource_name: str,
        indexer_name: str,
        storage_connection_string: str,
        container_name: str
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.index_name = index_name
        self.datasource_name = datasource_name
        self.indexer_name = indexer_name
        self.storage_connection_string = storage_connection_string
        self.container_name = container_name

        # SDKクライアント
        self.indexer_client = SearchIndexerClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key)
        )

    def delete_search_resources(self) -> Tuple[bool, List[str]]:
        """
        データソース・インデクサ・インデックスを削除。

        Returns:
            tuple[bool, List[str]]: (成功フラグ, エラー内容リスト)
        """
        errors: List[str] = []
        try:
            # --- インデクサ削除 ---
            try:
                self.indexer_client.delete_indexer(self.indexer_name)
                logger.info(f"Indexer '{self.indexer_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Indexer '{self.indexer_name}' not found.")
            except Exception as e:
                msg = f"Error deleting indexer: {str(e)}"
                logger.warning(msg)
                errors.append(msg)

            # --- インデックス削除 ---
            try:
                self.index_client.delete_index(self.index_name)
                logger.info(f"Index '{self.index_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Index '{self.index_name}' not found.")
            except Exception as e:
                msg = f"Error deleting index: {str(e)}"
                logger.warning(msg)
                errors.append(msg)

            # --- データソース削除 ---
            try:
                self.indexer_client.delete_data_source_connection(self.datasource_name)
                logger.info(f"Data source '{self.datasource_name}' deleted.")
            except ResourceNotFoundError:
                logger.info(f"Data source '{self.datasource_name}' not found.")
            except Exception as e:
                msg = f"Error deleting data source: {str(e)}"
                logger.warning(msg)
                errors.append(msg)

        except Exception as e:
            msg = f"Unexpected error while deleting resources: {str(e)}"
            logger.error(msg)
            errors.append(msg)

        return (len(errors) == 0, errors)

    def create_search_resources(self) -> bool:
        """
        データソース・インデックス・インデクサーを作成（存在すれば更新）し、
        インデクサーを実行して終了。

        Returns:
            bool: 作成と実行に成功したかどうか
        """
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }

            # 1. データソース作成/更新
            ds_url = f"{self.endpoint}/datasources/{self.datasource_name}?api-version={self.api_version}"
            ds_body = {
                "name": self.datasource_name,
                "type": "azureblob",
                "credentials": {
                    "connectionString": self.storage_connection_string
                },
                "container": {
                    "name": self.container_name
                }
            }
            ds_resp = requests.put(ds_url, headers=headers, json=ds_body)
            ds_resp.raise_for_status()
            logger.info(f"Data source '{self.datasource_name}' created/updated.")

            # 2. インデックス作成/更新
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
                                    {"fieldName": "content"}
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

            # 3. インデクサー作成/更新
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

            time.sleep(2)  # API 反映待ち

            # 4. インデクサーを今すぐ実行
            run_url = f"{self.endpoint}/indexers/{self.indexer_name}/run?api-version={self.api_version}"
            run_resp = requests.post(run_url, headers=headers)
            run_resp.raise_for_status()
            logger.info(f"Indexer '{self.indexer_name}' triggered to run.")

            return True
        except Exception as e:
            logger.error(f"Error creating search resources: {str(e)}")
            return False

    def run_indexer(self) -> bool:
        """インデクサーを手動実行"""
        try:
            headers = {
                'Content-Type': 'application/json',
                'api-key': self.api_key
            }
            url = f"{self.endpoint}/indexers/{self.indexer_name}/run?api-version={self.api_version}"
            resp = requests.post(url, headers=headers)
            if resp.status_code in self.SUCCESSFUL_RUN_STATUS_CODES:
                logger.info(f"Indexer '{self.indexer_name}' run triggered.")
                return True
            logger.warning(f"Failed to trigger indexer run: {resp.status_code}, {resp.text}")
            return False
        except Exception as e:
            logger.error(f"Error running indexer: {str(e)}")
            return False

    def get_indexer_status(self) -> str:
        """
        インデクサーの最新ステータスを返す ("success", "inProgress" 等)。
        - "not_found": インデクサが存在しない
        - "error:" で始まる: 何らかのエラー
        """
        try:
            idx_client = SearchIndexerClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            status = idx_client.get_indexer_status(self.indexer_name)
            return status.last_result.status
        except ResourceNotFoundError:
            return "not_found"
        except HttpResponseError as e:
            logger.error(f"Azure Search error: {str(e)}")
            return f"error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"error: {str(e)}"

    def search_pdf_content(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """クエリでPDFのコンテンツを検索。"""
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            search_opts = {
                "select": ["content", "metadata_storage_name", "metadata_storage_path"],
                "top": top_k,
                "query_type": "simple",
                "query_language": "ja-JP"
            }
            results = search_client.search(query, **search_opts)

            docs = []
            for result in results:
                fn = result.get("metadata_storage_name", "Unknown")
                ct = result.get("content", "")
                path = result.get("metadata_storage_path", "")
                if ct:
                    snippet = ct[:2000] + ("..." if len(ct) > 2000 else "")
                    docs.append({
                        "filename": fn,
                        "content": snippet,
                        "path": path
                    })
            return {"documents": docs, "count": len(docs), "query": query}
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"error": str(e), "documents": [], "count": 0, "query": query}

    async def get_all_pdf_content(self) -> Dict[str, Any]:
        """インデックス内のすべてのPDFコンテンツを取得 (query='*')。"""
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            results = search_client.search(
                search_text="*",
                select=["content", "metadata_storage_name", "metadata_storage_path"],
                top=100
            )
            docs = []
            for r in results:
                fn = r.get("metadata_storage_name", "Unknown")
                ct = r.get("content", "")
                path = r.get("metadata_storage_path", "")
                docs.append({
                    "filename": fn,
                    "content": ct,
                    "path": path
                })
            return {"documents": docs, "count": len(docs)}

        except Exception as e:
            logger.error(f"Error retrieving all PDF content: {str(e)}")
            return {"error": str(e), "documents": [], "count": 0}

    def clear_all_documents(self) -> bool:
        """
        インデックス内のすべてのドキュメントを削除するが、インデックスの構造は保持する。
        ワイルドカード（'*'）で一括削除を行う。
        """
        try:
            url = f"{self.endpoint}/indexes/{self.index_name}/docs/index?api-version={self.api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            payload = {
                "value": [
                    {
                        "@search.action": "delete",
                        "id": "*"
                    }
                ]
            }
            logger.info("Clearing all documents from the index using wildcard delete.")
            response = requests.post(url, headers=headers, json=payload)

            if response.status_code in (200, 207):
                logger.info("Index documents cleared successfully.")
                return True
            else:
                logger.error(f"Failed to clear index documents: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error clearing index documents: {str(e)}")
            return False

# =========================================================
# Azure OpenAI管理クラス
# =========================================================
class AzureOpenAIManager:
    """Azure OpenAI の操作をまとめたクラス。"""

    def __init__(self, endpoint: str, api_key: str, api_version: str, model_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.model_name = model_name

    def chat_completion(
        self,
        system_message: str,
        user_prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        Azure OpenAI ChatCompletion を実行する。

        Args:
            system_message (str): システムメッセージ
            user_prompt (str): ユーザーからのプロンプト
            temperature (float): 生成の多様性
            top_p (float): nucleus sampling

        Returns:
            str: モデルの返答文字列
        """
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
# FastAPIアプリ
# =========================================================
app = FastAPI(docs_url="/docs", redoc_url="/redoc")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================================================
# インスタンス生成
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
# エンドポイント定義
# =========================================================
@app.post("/upload", response_model=StandardResponse)
async def upload_pdf(file: UploadFile = File(...)) -> StandardResponse:
    """
    新しいPDFをアップロードするたびに検索リソース(インデックスなど)とBlobをリセットし、
    アップロードしたPDF一つだけがインデックスされるようにするエンドポイント。

    1. 既存のAzure Searchリソース削除
    2. 既存Blobを削除
    3. 新PDFをアップロード
    4. Searchリソースを再作成
    5. インデックス内のドキュメントをクリア
    6. インデクサを手動実行
    """
    try:
        logger.info("Deleting existing search resources...")
        success, errors = search_manager.delete_search_resources()
        if not success:
            logger.warning(f"Some errors occurred while deleting search resources: {errors}")

        logger.info("Deleting all existing blobs...")
        blob_manager.delete_all_blobs()

        logger.info("Uploading new PDF...")
        content = await file.read()
        blob_manager.upload_blob(file.filename, content)

        logger.info("Creating new search resources...")
        created = search_manager.create_search_resources()
        if not created:
            return StandardResponse(
                success=False,
                message="Failed to recreate search resources.",
                timestamp=datetime.now().isoformat()
            )

        # clear_all_documents を呼び出す
        logger.info("Clearing all existing documents from the new index...")
        cleared = search_manager.clear_all_documents()
        if not cleared:
            logger.warning("Failed to clear documents in the index (it may already be empty).")

        logger.info("Manually triggering indexer run...")
        run_res = search_manager.run_indexer()
        indexer_status = "running" if run_res else "not_running"

        return StandardResponse(
            success=True,
            message="ファイルアップロードとインデックス再作成が完了しました。既存のドキュメントは削除済みです。",
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


@app.get("/indexer-status", response_model=StandardResponse)
async def get_indexer_status() -> StandardResponse:
    """
    インデクサーのステータスを返すエンドポイント。
    """
    try:
        status = search_manager.get_indexer_status()
        if status == "not_found":
            return StandardResponse(
                success=False,
                message="インデクサーが見つかりません。PDFをアップロードしてください。",
                data={"indexer_status": "not_found"},
                timestamp=datetime.now().isoformat()
            )
        if status.startswith("error:"):
            return StandardResponse(
                success=False,
                message=f"インデクサー エラー: {status}",
                data={"indexer_status": status},
                timestamp=datetime.now().isoformat()
            )

        return StandardResponse(
            success=True,
            message="インデクサーの状態を取得しました。",
            data={"indexer_status": status},
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Failed to get indexer status: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Failed to get indexer status: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


@app.get("/pdf-content", response_model=StandardResponse)
async def get_pdf_content_endpoint(query: str = "*") -> StandardResponse:
    """
    指定したクエリ(query)をもとにAzure Searchへ全文検索を行い、
    ヒットしたPDFのコンテンツを返す。
    """
    try:
        status = search_manager.get_indexer_status()
        if status == "not_found":
            return StandardResponse(
                success=False,
                message="インデクサーが見つかりません。PDFをアップロードしてください。",
                data={"status": "not_found"},
                timestamp=datetime.now().isoformat()
            )
        if status.startswith("error"):
            return StandardResponse(
                success=False,
                message=f"インデクサー状態エラー: {status}",
                data={"status": "error"},
                timestamp=datetime.now().isoformat()
            )
        if status != "success":
            return StandardResponse(
                success=False,
                message="まだインデクシング中かもしれません。少し待って再試行してください。",
                data={"status": status},
                timestamp=datetime.now().isoformat()
            )

        results = search_manager.search_pdf_content(query)
        return StandardResponse(
            success=True,
            message="PDFの検索結果を返します。",
            data=results,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in get_pdf_content_endpoint: {str(e)}")
        return StandardResponse(
            success=False,
            message=f"Failed to retrieve PDF content: {str(e)}",
            timestamp=datetime.now().isoformat()
        )


@app.post("/chat", response_model=StandardResponse)
async def chat(prompt: str) -> StandardResponse:
    """
    アップロード済みのPDF内容をもとにAzure OpenAIチャット回答を行う。
    """
    try:
        status = search_manager.get_indexer_status()
        if status == "not_found":
            return StandardResponse(
                success=False,
                message="インデクサーが見つかりません。先にPDFをアップロードしてください。",
                timestamp=datetime.now().isoformat()
            )
        if status.startswith("error:"):
            return StandardResponse(
                success=False,
                message=f"インデクサー状態エラー: {status}",
                timestamp=datetime.now().isoformat()
            )
        if status != "success":
            return StandardResponse(
                success=False,
                message="まだインデクシングが完了していません。再度お試しください。",
                data={"status": status},
                timestamp=datetime.now().isoformat()
            )

        # PDFの全コンテンツを取得
        pdf_results = await search_manager.get_all_pdf_content()
        docs = pdf_results.get("documents", [])
        pdf_context = ""
        sources = []

        if docs:
            for i, doc in enumerate(docs):
                pdf_context += f"Document {i+1}: {doc['filename']}\n"
                pdf_context += f"Content: {doc['content']}\n\n"
                sources.append({"filename": doc['filename'], "path": doc['path']})
        else:
            pdf_context = "No PDF content found."

        # システムメッセージ
        system_message = (
            "You are an AI assistant that helps answer questions based on PDF documents.\n"
            "Answer based ONLY on the content in the documents provided below.\n"
            "If the information isn't in the documents, clearly state that.\n\n"
            "Here is the content from the uploaded PDF documents:\n\n"
            f"{pdf_context}"
        )

        # OpenAIに問い合わせ
        answer = openai_manager.chat_completion(system_message, prompt)

        return StandardResponse(
            success=True,
            message="チャット応答が完了しました。",
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
async def health() -> Dict[str, str]:
    """
    ヘルスチェック用。
    """
    return {"status": "healthy", "service": "simplerag"}


# # ローカル実行用
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     host = os.environ.get("HOST", "0.0.0.0")
#     logger.info(f"Starting FastAPI server on {host}:{port}...")
#     uvicorn.run(app, host=host, port=port)
