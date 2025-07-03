# SimpleRAG Backend

PDF 문서 기반 RAG(Retrieval-Augmented Generation) API 서버입니다.

## 주요 기능

- PDF 파일 업로드 및 Azure Blob Storage 저장
- Azure Cognitive Search를 통한 PDF 내용 인덱싱
- Azure OpenAI를 통한 문서 기반 질의응답
- 업로드 상태 모니터링
- 전문 검색 기능

## 환경 변수

다음 환경 변수들을 설정해야 합니다:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
CONTAINER_NAME=your_container_name

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_API_VERSION=2023-11-01
INDEX_NAME=azureblob-index
DATASOURCE_NAME=simplerag
INDEXER_NAME=azureblob-indexer

# Azure OpenAI
AZURE_OPENAI_API_ENDPOINT=https://your-openai-service.openai.azure.com/
AZURE_OPENAI_API_KEY=your_openai_key
AZURE_OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_MODEL_NAME=gpt-4

# 보안 설정 (선택사항)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
LOG_TO_FILE=true
```

## API 엔드포인트

### 핵심 엔드포인트

- `POST /upload` - PDF 파일 업로드
- `POST /chat` - 문서 기반 질의응답
- `GET /pdf-content` - PDF 내용 검색
- `GET /upload-status` - 업로드 상태 확인
- `GET /indexer-status` - 인덱서 상태 확인
- `POST /index-reset` - 인덱스 리셋
- `GET /health` - 헬스 체크

### 실행 방법

```bash
# 의존성 설치
pip install -r requirements.txt

# 서버 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 주요 개선사항

### 보안 강화
- CORS 설정 개선 (와일드카드 제거)
- 파일 업로드 검증 강화
- 입력값 검증 추가

### 에러 처리 개선
- 일관된 에러 응답 형식
- 전역 예외 처리
- 상세한 로깅

### 성능 최적화
- 요청 로깅 미들웨어
- 파일 크기 제한
- 응답 시간 측정

### 코드 품질 향상
- 유틸리티 함수 분리
- 상수 정의
- 한국어 메시지 통일

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.