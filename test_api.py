"""
Quick test script for your Ayurvedic RAG API
Run locally or against deployed Render URL
"""

import requests
import json
import time
from typing import Dict, Any

# Change this to your Render URL after deployment
BASE_URL = "http://localhost:8000"  # For local testing
# BASE_URL = "https://your-app.onrender.com"  # For production

def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("🏥 Testing Health Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Status:", data.get("status"))
        print("✅ Model Loaded:", data.get("model_loaded"))
        print("✅ Groq Model:", data.get("groq_model"))
        print("✅ Active Sessions:", data.get("active_sessions"))
        
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_chat(query: str, session_id: str = None):
    """Test chat endpoint"""
    print("\n" + "="*70)
    print(f"💬 Testing Chat: {query}")
    print("="*70)
    
    payload = {
        "query": query,
        "use_memory": True
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start
        
        response.raise_for_status()
        data = response.json()
        
        print(f"\n📊 Session ID: {data['session_id']}")
        print(f"🎯 Intent: {data['intent']}")
        print(f"🏷️  Entities: {data['entities']}")
        print(f"⚡ Response Time: {elapsed:.2f}s")
        print(f"📈 Confidence: {data['confidence']:.1%}")
        
        print(f"\n💡 Answer:\n{data['answer']}")
        
        print(f"\n📚 Sources ({len(data['sources'])}):")
        for i, source in enumerate(data['sources'][:3], 1):
            print(f"  [{i}] {source['source']} (Page {source['page']}) - Score: {source['score']:.3f}")
        
        print(f"\n🔍 Reasoning Steps:")
        for step in data['reasoning']:
            print(f"  • {step}")
        
        return data['session_id']
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout - API might be cold starting (try again)")
        return None
    except Exception as e:
        print(f"❌ Chat failed: {e}")
        return None


def test_streaming(query: str):
    """Test streaming endpoint"""
    print("\n" + "="*70)
    print(f"📡 Testing Streaming: {query}")
    print("="*70)
    
    payload = {
        "query": query,
        "use_memory": False
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=60
        )
        
        print("\n💬 Streaming Response:")
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    token = decoded[6:]  # Remove 'data: ' prefix
                    if token == '[DONE]':
                        print("\n✅ Stream complete")
                        break
                    elif token.startswith('[ERROR]'):
                        print(f"\n❌ {token}")
                        break
                    else:
                        print(token, end='', flush=True)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Streaming failed: {e}")
        return False


def test_conversation_flow():
    """Test multi-turn conversation with memory"""
    print("\n" + "="*70)
    print("🔄 Testing Conversation Flow (Memory)")
    print("="*70)
    
    queries = [
        "What is Pitta dosha?",
        "What are its symptoms?",  # Should use memory context
        "How can I balance it?",   # Should use memory context
    ]
    
    session_id = None
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Turn {i} ---")
        session_id = test_chat(query, session_id)
        
        if not session_id:
            print("❌ Conversation flow failed")
            return False
        
        time.sleep(1)  # Brief pause between requests
    
    print("\n✅ Conversation flow complete")
    return True


def test_stats():
    """Test stats endpoint"""
    print("\n" + "="*70)
    print("📊 Testing Stats Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f"{BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print("Total Sessions:", data.get("total_sessions"))
        print("Total Messages:", data.get("total_messages"))
        print("Avg Messages/Session:", data.get("avg_messages_per_session"))
        print("LLM Provider:", data.get("llm_provider"))
        print("Model:", data.get("model"))
        
        return True
    except Exception as e:
        print(f"❌ Stats failed: {e}")
        return False


def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print("🧪 AYURVEDIC RAG API TEST SUITE")
    print("="*70)
    print(f"Target: {BASE_URL}")
    
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health()))
    time.sleep(1)
    
    # Test 2: Simple chat
    results.append(("Simple Chat", bool(test_chat("What is Agni in Ayurveda?"))))
    time.sleep(1)
    
    # Test 3: Conversation flow
    results.append(("Conversation Flow", test_conversation_flow()))
    time.sleep(1)
    
    # Test 4: Streaming
    results.append(("Streaming", test_streaming("Explain Vata dosha briefly")))
    time.sleep(1)
    
    # Test 5: Stats
    results.append(("Stats", test_stats()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working perfectly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
        print(f"Using custom URL: {BASE_URL}")
    
    run_all_tests()