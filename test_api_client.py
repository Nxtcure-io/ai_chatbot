"""
Test client for Clinical Trial RAG Chatbot API
Demonstrates how to use the API service
"""
import requests
import json
import time


class ChatbotAPIClient:
    """Client for interacting with the Chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client
        
        Args:
            base_url: Base URL of the API service
        """
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error checking health: {e}")
            return None
    
    def chat(self, query: str, n_results: int = 5):
        """
        Send chat query to API
        
        Args:
            query: User question
            n_results: Number of results to retrieve
            
        Returns:
            API response with answer and sources
        """
        try:
            payload = {
                "query": query,
                "n_results": n_results
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error sending query: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            return None
    
    def get_stats(self):
        """Get API statistics"""
        try:
            response = requests.get(f"{self.base_url}/stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting stats: {e}")
            return None
    
    def reset_stats(self):
        """Reset API statistics"""
        try:
            response = requests.post(f"{self.base_url}/reset-stats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error resetting stats: {e}")
            return None


def main():
    """Test the API with sample queries"""
    print("="*70)
    print("Clinical Trial RAG Chatbot API - Test Client")
    print("="*70)
    
    # Initialize client
    client = ChatbotAPIClient()
    
    # Health check
    print("\n1. Health Check")
    print("-"*70)
    health = client.health_check()
    if health:
        print(f"Status: {health['status']}")
        print(f"Service: {health['service']}")
        print(f"Chatbot Loaded: {health['chatbot_loaded']}")
    else:
        print("❌ API is not available. Make sure the API server is running.")
        print("   Run: python api.py")
        return
    
    # Test queries
    test_queries = [
        "What is the current recruitment status of trial NCT05812131?",
        "Are there any clinical trials for PTSD?",
        "Which trials accept healthy volunteers?"
    ]
    
    print("\n2. Test Queries")
    print("-"*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-"*70)
        
        result = client.chat(query)
        
        if result and result.get('success'):
            print(f"\n✓ Success!")
            print(f"\nAnswer:\n{result['answer']}")
            
            print(f"\nSources ({len(result['sources'])}):")
            for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                print(f"  {j}. {source['NCTId']}: {source['Title']}")
                print(f"     Relevance: {source['Relevance']}")
            
            print(f"\nTiming:")
            print(f"  Retrieval: {result['timing']['retrieval_time']}")
            print(f"  API Call: {result['timing']['api_time']}")
            print(f"  Total: {result['timing']['total_time']}")
        else:
            print("❌ Query failed")
        
        # Pause between queries
        if i < len(test_queries):
            time.sleep(1)
    
    # Get statistics
    print("\n3. API Statistics")
    print("-"*70)
    stats = client.get_stats()
    if stats:
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Avg Retrieval Time: {stats['avg_retrieval_time']}")
        print(f"Avg API Time: {stats['avg_api_time']}")
        print(f"Avg Total Time: {stats['avg_total_time']}")
    
    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)


def test_curl_examples():
    """Print cURL examples for testing"""
    print("\n" + "="*70)
    print("cURL Examples")
    print("="*70)
    
    print("\n1. Health Check:")
    print("curl http://localhost:8000/health")
    
    print("\n2. Chat Query:")
    print("""curl -X POST http://localhost:8000/chat \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Are there any clinical trials for PTSD?", "n_results": 5}'""")
    
    print("\n3. Get Statistics:")
    print("curl http://localhost:8000/stats")
    
    print("\n4. Reset Statistics:")
    print("curl -X POST http://localhost:8000/reset-stats")
    
    print("\n5. API Documentation:")
    print("Open in browser: http://localhost:8000/docs")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--curl":
        test_curl_examples()
    else:
        main()
        print("\nTo see cURL examples, run:")
        print("  python test_api_client.py --curl")

