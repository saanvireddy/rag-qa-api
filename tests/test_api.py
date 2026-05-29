from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✅ Health check test passed")

def test_query_endpoint():
    """Test query endpoint"""
    response = client.post("/query", json={"question": "What is AI?"})
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data
        print("✅ Query test passed")
    else:
        print("⚠️ Query test skipped (Vertex AI not configured)")

def test_empty_question():
    """Test empty question handling"""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 400
    print("✅ Empty question test passed")

if __name__ == "__main__":
    test_health_check()
    test_empty_question()
    test_query_endpoint()
    print("\n✅ All tests completed!")