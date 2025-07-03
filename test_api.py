#!/usr/bin/env python3
"""
SimpleRAG API 기본 테스트 스크립트
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"헬스 체크: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"헬스 체크 실패: {e}")
        return False

def test_upload_status():
    """업로드 상태 확인 테스트"""
    try:
        response = requests.get(f"{BASE_URL}/upload-status", timeout=10)
        print(f"업로드 상태: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"업로드 상태 확인 실패: {e}")
        return False

def test_indexer_status():
    """인덱서 상태 확인 테스트"""
    try:
        response = requests.get(f"{BASE_URL}/indexer-status", timeout=10)
        print(f"인덱서 상태: {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        return response.status_code == 200
    except Exception as e:
        print(f"인덱서 상태 확인 실패: {e}")
        return False

def main():
    print("SimpleRAG API 테스트 시작...")
    print("=" * 50)
    
    tests = [
        ("헬스 체크", test_health),
        ("업로드 상태", test_upload_status),
        ("인덱서 상태", test_indexer_status)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name} 테스트:")
        print("-" * 30)
        result = test_func()
        results.append((name, result))
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("테스트 결과:")
    for name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{name}: {status}")
    
    success_count = sum(1 for _, result in results if result)
    print(f"\n총 {len(results)}개 테스트 중 {success_count}개 성공")

if __name__ == "__main__":
    main()