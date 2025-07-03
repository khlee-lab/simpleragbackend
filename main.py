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
# 로그 설정
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simplerag.log') if os.environ.get('LOG_TO_FILE') else logging.NullHandler()
    ]
)
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
AZURE_SEARCH_API_VERSION = os.environ.get("AZURE_SEARCH_API_VERSION", "2023-11-01")

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

class ChatRequest(BaseModel):
    """チャットリクエスト用モデル。"""
    prompt: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    
    class Config:
        str_strip_whitespace = True
        min_anystr_length = 1
        max_anystr_length = 10000

class SearchRequest(BaseModel):
    """検索リクエスト用モデル。"""
    query: str = "*"
    top_k: Optional[int] = 3
    
    class Config:
        str_strip_whitespace = True
        min_anystr_length = 1
        max_anystr_length = 1000

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
            credential=AzureKeyCredential(self.api_key),
            api_version=self.api_version
        )
        self.index_client = SearchIndexClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=self.api_version
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

            # 2. インデックス作成/更新 (semantic機能なし)
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
                ]
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
            if run_resp.status_code in self.SUCCESSFUL_RUN_STATUS_CODES:
                logger.info(f"Indexer '{self.indexer_name}' triggered to run.")
            elif run_resp.status_code == 409:
                # 既に実行中の場合
                logger.warning("Indexer is already running. (409 Conflict)")
            else:
                run_resp.raise_for_status()

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
            elif resp.status_code == 409:
                logger.warning("Indexer is already running. (409 Conflict)")
                return True
            else:
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
            status = self.indexer_client.get_indexer_status(self.indexer_name)
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
        インデックス内のすべてのドキュメントを削除するが、インデックス構造は保持する。
        ドキュメントを個別に削除 (ワイルドカード '*' は使わない)
        """
        try:
            search_client = SearchClient(
                endpoint=self.endpoint,
                index_name=self.index_name,
                credential=AzureKeyCredential(self.api_key),
                api_version=self.api_version
            )
            # 1. 全件検索
            all_docs = list(search_client.search(search_text="*", top=1000))
            if not all_docs:
                logger.info("No documents found; nothing to delete.")
                return True

            # 2. 削除バッチ作成
            actions = []
            for doc in all_docs:
                doc_id = doc.get("id")
                if doc_id:
                    actions.append({
                        "@search.action": "delete",
                        "id": doc_id
                    })

            if not actions:
                logger.info("No valid document IDs found; skip delete.")
                return True

            url = f"{self.endpoint}/indexes/{self.index_name}/docs/index?api-version={self.api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            payload = {"value": actions}

            logger.info(f"Deleting {len(actions)} documents from the index individually...")
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code in (200, 207):
                logger.info("Index documents cleared successfully.")
                return True
            else:
                logger.error(f"Failed to delete docs individually: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Error clearing index documents: {str(e)}")
            return False

    def delete_index_and_indexer_only(self) -> List[str]:
        """
        インデクサとインデックスだけを削除する（データソースは残す）。
        Returns:
            List[str]: 削除時に発生したエラーメッセージのリスト（なければ空）
        """
        errors: List[str] = []
        # インデクサ削除
        try:
            self.indexer_client.delete_indexer(self.indexer_name)
            logger.info(f"Indexer '{self.indexer_name}' deleted.")
        except ResourceNotFoundError:
            logger.info(f"Indexer '{self.indexer_name}' not found.")
        except Exception as e:
            msg = f"Error deleting indexer: {str(e)}"
            logger.warning(msg)
            errors.append(msg)

        # インデックス削除
        try:
            self.index_client.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted.")
        except ResourceNotFoundError:
            logger.info(f"Index '{self.index_name}' not found.")
        except Exception as e:
            msg = f"Error deleting index: {str(e)}"
            logger.warning(msg)
            errors.append(msg)

        return errors

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
# 상수 정의
# =========================================================
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = {"application/pdf", "application/x-pdf"}
ALLOWED_FILE_EXTENSIONS = {".pdf"}

# =========================================================
# 유틸리티 함수
# =========================================================
def validate_pdf_file(file: UploadFile) -> Optional[str]:
    """PDF 파일 유효성 검사"""
    if not file.filename:
        return "파일명이 없습니다."
    
    if not file.filename.lower().endswith('.pdf'):
        return "PDF 파일만 업로드 가능합니다."
    
    if file.content_type not in ALLOWED_FILE_TYPES:
        return f"지원되지 않는 파일 형식입니다. PDF 파일만 업로드 가능합니다."
    
    return None

def create_error_response(message: str, data: Optional[Any] = None) -> StandardResponse:
    """에러 응답 생성"""
    return StandardResponse(
        success=False,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat()
    )

def create_success_response(message: str, data: Optional[Any] = None) -> StandardResponse:
    """성공 응답 생성"""
    return StandardResponse(
        success=True,
        message=message,
        data=data,
        timestamp=datetime.now().isoformat()
    )

# =========================================================
# 에러 처리 상수
# =========================================================
ERROR_MESSAGES = {
    "UPLOAD_FAILED": "파일 업로드에 실패했습니다",
    "INVALID_FILE": "유효하지 않은 파일입니다",
    "FILE_TOO_LARGE": "파일 크기가 너무 큽니다",
    "INDEXER_NOT_FOUND": "인덱서를 찾을 수 없습니다",
    "SEARCH_FAILED": "검색에 실패했습니다",
    "CHAT_FAILED": "채팅 응답 생성에 실패했습니다",
    "VALIDATION_ERROR": "입력값이 유효하지 않습니다"
}

# FastAPI 애플리케이션 및 미들웨어 설정
app = FastAPI(
    title="SimpleRAG API",
    description="PDF 문서 기반 RAG(Retrieval-Augmented Generation) API",
    version="1.0.0",
    docs_url="/docs", 
    redoc_url="/redoc"
)

# 에러 핸들러 추가
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"전역 예외 발생: {str(exc)}")
    return create_error_response("서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.")

# 요청 로깅 미들웨어
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url} - {response.status_code} - {process_time:.2f}s")
    return response

# CORS設定 - 보안 강화
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"]
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
    새로운 PDF를 업로드할 때마다 검색 리소스(인덱스 등)와 Blob을 리셋하고,
    업로드한 PDF만 인덱싱되도록 하는 엔드포인트.

    1. 파일 유효성 검사
    2. 기존 Azure Search 리소스 삭제
    3. 기존 Blob 삭제
    4. 새 PDF 업로드
    5. Search 리소스 재생성
    6. 인덱스 내 문서 삭제
    7. 인덱서 수동 실행
    """
    try:
        # 1. 파일 유효성 검사
        validation_error = validate_pdf_file(file)
        if validation_error:
            return create_error_response(validation_error)
        
        # 파일 크기 검사 (스트림으로 읽기 전에 미리 확인)
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            return create_error_response(f"파일 크기가 너무 큽니다. 최대 {MAX_FILE_SIZE // (1024*1024)}MB까지 업로드 가능합니다.")
        
        if len(content) == 0:
            return create_error_response("빈 파일입니다.")

        # 2. 기존 Azure Search 리소스 삭제
        logger.info("기존 검색 리소스 삭제 중...")
        success, errors = search_manager.delete_search_resources()
        if not success:
            logger.warning(f"검색 리소스 삭제 중 일부 오류 발생: {errors}")

        # 3. 기존 Blob 삭제
        logger.info("기존 Blob 삭제 중...")
        blob_manager.delete_all_blobs()

        # 4. 새 PDF 업로드
        logger.info("새 PDF 업로드 중...")
        blob_manager.upload_blob(file.filename, content)

        # 5. 리소스 재생성 (DataSource, Index, Indexer) & 인덱서 실행
        logger.info("새 검색 리소스 생성 중...")
        created = search_manager.create_search_resources()
        if not created:
            return create_error_response("검색 리소스 재생성에 실패했습니다.")

        # 6. 인덱스 내 문서 개별 삭제
        logger.info("인덱스에서 기존 문서 삭제 중...")
        cleared = search_manager.clear_all_documents()
        if not cleared:
            logger.warning("인덱스 문서 삭제 실패 (이미 비어있을 수 있음)")

        # 7. 인덱서 수동 실행 (2회차)
        logger.info("인덱서 수동 실행 중...")
        run_res = search_manager.run_indexer()
        indexer_status = "running" if run_res else "not_running"

        return create_success_response(
            "파일 업로드와 인덱스 재생성이 완료되었습니다. 기존 문서는 삭제되었습니다.",
            {"filename": file.filename, "indexer_status": indexer_status, "file_size": len(content)}
        )

    except Exception as e:
        logger.error(f"업로드 오류: {str(e)}")
        return create_error_response(f"업로드 실패: {str(e)}")

@app.post("/index-reset", response_model=StandardResponse)
def index_reset() -> StandardResponse:
    """
    인덱서와 인덱스를 삭제하고 재생성(리셋)하는 엔드포인트.
    ※ 데이터 소스는 삭제하지 않음
    """
    try:
        logger.info("인덱서와 인덱스만 삭제 중 (데이터 소스는 유지)")
        errors = search_manager.delete_index_and_indexer_only()
        if errors:
            logger.warning(f"인덱서/인덱스 삭제 중 일부 오류 발생: {errors}")

        logger.info("인덱스와 인덱서 재생성 중...")
        success = search_manager.create_search_resources()
        if not success:
            return create_error_response("인덱스/인덱서 재생성에 실패했습니다.")

        return create_success_response(
            "인덱스와 인덱서를 리셋했습니다.",
            {"reset_time": datetime.now().isoformat()}
        )

    except Exception as e:
        logger.error(f"인덱스 리셋 오류: {str(e)}")
        return create_error_response(f"인덱스 리셋 실패: {str(e)}")

@app.get("/indexer-status", response_model=StandardResponse)
async def get_indexer_status() -> StandardResponse:
    """
    인덱서의 상태를 반환하는 엔드포인트.
    """
    try:
        status = search_manager.get_indexer_status()
        
        if status == "not_found":
            return create_error_response(
                "인덱서를 찾을 수 없습니다. PDF를 업로드해주세요.",
                {"indexer_status": "not_found"}
            )
        
        if status.startswith("error:"):
            return create_error_response(
                f"인덱서 오류: {status}",
                {"indexer_status": status}
            )

        return create_success_response(
            "인덱서 상태를 가져왔습니다.",
            {"indexer_status": status}
        )

    except Exception as e:
        logger.error(f"인덱서 상태 가져오기 실패: {str(e)}")
        return create_error_response(f"인덱서 상태 가져오기 실패: {str(e)}")

@app.get("/pdf-content", response_model=StandardResponse)
async def get_pdf_content_endpoint(query: str = "*", top_k: int = 3) -> StandardResponse:
    """
    지정된 쿼리로 Azure Search에서 전문 검색을 수행하고,
    일치하는 PDF 콘텐츠를 반환합니다.
    """
    try:
        # 입력 검증
        if len(query.strip()) == 0:
            return create_error_response("검색어를 입력해주세요.")
        
        if len(query) > 1000:
            return create_error_response("검색어가 너무 깁니다. 1000자 이하로 입력해주세요.")
        
        if top_k < 1 or top_k > 50:
            return create_error_response("결과 개수는 1~50개 사이여야 합니다.")
        
        # 인덱서 상태 확인
        status = search_manager.get_indexer_status()
        if status == "not_found":
            return create_error_response("인덱서를 찾을 수 없습니다. PDF를 업로드해주세요.", {"status": "not_found"})
        
        if status.startswith("error"):
            return create_error_response(f"인덱서 상태 오류: {status}", {"status": "error"})
        
        if status != "success":
            return create_error_response("아직 인덱싱 중일 수 있습니다. 잠시 후 다시 시도해주세요.", {"status": status})

        # PDF 내용 검색
        results = search_manager.search_pdf_content(query, top_k)
        
        if "error" in results:
            return create_error_response(f"검색 실패: {results['error']}")
        
        return create_success_response(
            "PDF 검색 결과를 반환합니다.",
            {
                **results,
                "top_k": top_k
            }
        )

    except Exception as e:
        logger.error(f"PDF 내용 검색 오류: {str(e)}")
        return create_error_response(f"PDF 내용 검색 실패: {str(e)}")

@app.post("/chat", response_model=StandardResponse)
async def chat(request: ChatRequest) -> StandardResponse:
    """
    업로드된 PDF 내용을 기반으로 Azure OpenAI 채팅 응답을 수행합니다.
    """
    try:
        # 입력 검증
        if not request.prompt or len(request.prompt.strip()) == 0:
            return create_error_response("질문을 입력해주세요.")
        
        if len(request.prompt) > 10000:
            return create_error_response("질문이 너무 깁니다. 10000자 이하로 입력해주세요.")
        
        # 인덱서 상태 확인
        status = search_manager.get_indexer_status()
        if status == "not_found":
            return create_error_response("인덱서를 찾을 수 없습니다. 먼저 PDF를 업로드해주세요.")
        
        if status.startswith("error:"):
            return create_error_response(f"인덱서 상태 오류: {status}")
        
        if status != "success":
            return create_error_response("아직 인덱싱이 완료되지 않았습니다. 잠시 후 다시 시도해주세요.", {"status": status})

        # PDF 전체 내용 검색
        pdf_results = await search_manager.get_all_pdf_content()
        docs = pdf_results.get("documents", [])
        
        if not docs:
            return create_error_response("PDF 내용을 찾을 수 없습니다.")
        
        # 컨텍스트 구성
        pdf_context = ""
        sources = []
        
        for i, doc in enumerate(docs):
            pdf_context += f"문서 {i+1}: {doc['filename']}\n"
            pdf_context += f"내용: {doc['content'][:5000]}...\n\n"  # 내용 길이 제한
            sources.append({"filename": doc['filename'], "path": doc['path']})
        
        # 시스템 메시지 구성
        system_message = (
            "당신은 PDF 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.\n"
            "아래 제공된 문서 내용만을 바탕으로 답변해주세요.\n"
            "문서에 없는 정보라면 명확히 언급해주세요.\n"
            "답변은 한국어로 해주세요.\n\n"
            "업로드된 PDF 문서 내용:\n\n"
            f"{pdf_context}"
        )

        # OpenAI API 호출
        answer = openai_manager.chat_completion(
            system_message, 
            request.prompt,
            temperature=request.temperature,
            top_p=request.top_p
        )

        return create_success_response(
            "채팅 응답이 완료되었습니다.",
            {
                "answer": answer, 
                "sources": sources,
                "document_count": len(docs),
                "prompt": request.prompt
            }
        )

    except Exception as e:
        logger.error(f"채팅 오류: {str(e)}")
        return create_error_response(f"채팅 실패: {str(e)}")

@app.get("/health")
async def health() -> Dict[str, Any]:
    """헬스 체크용 엔드포인트."""
    try:
        # 기본 서비스 정보
        health_info = {
            "status": "healthy",
            "service": "simplerag",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        # 환경 변수 확인 (민감한 정보는 제외)
        required_env_vars = [
            "AZURE_STORAGE_CONNECTION_STRING",
            "CONTAINER_NAME", 
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_KEY",
            "AZURE_OPENAI_API_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_MODEL_NAME"
        ]
        
        env_status = {}
        for var in required_env_vars:
            env_status[var] = "configured" if os.environ.get(var) else "missing"
        
        health_info["environment"] = env_status
        
        # 기본 연결 상태 확인
        try:
            # Blob 컨테이너 확인
            blob_manager.container_client.get_container_properties()
            health_info["blob_storage"] = "connected"
        except Exception as e:
            health_info["blob_storage"] = f"error: {str(e)}"
        
        try:
            # 인덱서 상태 확인 (간단한 체크)
            search_manager.get_indexer_status()
            health_info["search_service"] = "connected"
        except Exception as e:
            health_info["search_service"] = f"error: {str(e)}"
        
        return health_info
        
    except Exception as e:
        logger.error(f"헬스 체크 중 오류: {str(e)}")
        return {
            "status": "error",
            "service": "simplerag",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/upload-status", response_model=StandardResponse)
async def get_upload_status() -> StandardResponse:
    """
    PDF 파일의 업로드 상태를 확인하는 엔드포인트.
    - 컨테이너에 PDF가 있는지
    - 인덱서의 상태
    - 인덱스의 상태
    를 종합적으로 반환합니다.
    """
    try:
        # 1. Blob 컨테이너에 파일이 존재하는지 확인
        blobs = list(blob_manager.container_client.list_blobs())
        pdf_files = [blob.name for blob in blobs if blob.name.lower().endswith('.pdf')]
        
        # 2. 인덱서 상태 가져오기
        indexer_status = search_manager.get_indexer_status()
        
        # 3. 인덱스에 문서가 있는지 확인
        docs_count = 0
        try:
            search_client = SearchClient(
                endpoint=search_manager.endpoint,
                index_name=search_manager.index_name,
                credential=AzureKeyCredential(search_manager.api_key),
                api_version=search_manager.api_version
            )
            results = list(search_client.search("*", top=1))
            docs_count = len(results)
        except Exception as e:
            logger.warning(f"인덱스 문서 확인 중 오류: {str(e)}")
        
        # 상태 데이터 구성
        status_data = {
            "pdf_files": pdf_files,
            "pdf_count": len(pdf_files),
            "indexer_status": indexer_status,
            "has_documents": docs_count > 0,
            "documents_count": docs_count,
            "last_checked": datetime.now().isoformat()
        }
        
        # 4. 종합적인 상태 판정
        if not pdf_files:
            message = "PDF 파일이 아직 업로드되지 않았습니다."
            success = False
        elif indexer_status == "not_found":
            message = "PDF는 업로드되었지만 인덱서를 찾을 수 없습니다."
            success = False
        elif indexer_status.startswith("error"):
            message = f"PDF는 업로드되었지만 인덱서에 오류가 있습니다: {indexer_status}"
            success = False
        elif indexer_status == "inProgress":
            message = "PDF가 업로드되었고 인덱스 생성이 현재 진행 중입니다."
            success = True
        elif indexer_status == "success" and docs_count > 0:
            message = "PDF 업로드와 인덱스 생성이 완료되었습니다."
            success = True
        elif indexer_status == "success" and docs_count == 0:
            message = "PDF는 업로드되고 인덱서는 성공했지만 인덱스에 문서가 없습니다."
            success = False
        else:
            message = f"PDF 업로드 상태: 인덱서={indexer_status}, 문서수={docs_count}"
            success = True
            
        return StandardResponse(
            success=success,
            message=message,
            data=status_data,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"업로드 상태 확인 중 오류: {str(e)}")
        return create_error_response(f"업로드 상태 확인 중 오류가 발생했습니다: {str(e)}")

# ローカル実行用
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "localhost")

    logger.info("=== SimpleRAG API Server ===")
    logger.info(f"API documentation available at: http://{host}:{port}/docs")
    logger.info(f"Health check endpoint: http://{host}:{port}/health")
    logger.info(f"Starting FastAPI server on {host}:{port}...")

    uvicorn.run(app, host=host, port=port, log_level="info", reload=False)
