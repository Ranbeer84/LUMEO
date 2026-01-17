"""
Phase 4 Testing Script
Tests LLM integration and chat functionality
"""

import requests
import json
import time
from typing import Dict, List

# Configuration
BASE_URL = "http://localhost:5002/api"


class Phase4Tester:
    """Test suite for Phase 4 LLM integration"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
        self.conversation_id = None
    
    def test_ollama_connection(self):
        """Test if Ollama is running"""
        print("\n" + "="*60)
        print("TEST 1: Ollama Connection")
        print("="*60)
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                print(f"✓ Ollama is running")
                print(f"  Available models:")
                for model in models:
                    print(f"    - {model['name']}")
                
                if models:
                    print(f"\n✓ Ollama connection test passed")
                    self.test_results.append(("Ollama Connection", True))
                else:
                    print(f"\n⚠  Ollama running but no models installed")
                    print("  Run: ollama pull llama3.2:3b")
                    self.test_results.append(("Ollama Connection", False))
            else:
                print(f"✗ Ollama returned status {response.status_code}")
                self.test_results.append(("Ollama Connection", False))
                
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to Ollama at http://localhost:11434")
            print("  Make sure Ollama is running: ollama serve")
            self.test_results.append(("Ollama Connection", False))
        except Exception as e:
            print(f"✗ Ollama connection test failed: {str(e)}")
            self.test_results.append(("Ollama Connection", False))
    
    def test_llm_status(self):
        """Test LLM service status endpoint"""
        print("\n" + "="*60)
        print("TEST 2: LLM Service Status")
        print("="*60)
        
        try:
            response = requests.get(f"{self.base_url}/llm/status", timeout=10)
            data = response.json()
            
            if data.get('success'):
                print(f"✓ LLM service is active")
                print(f"  Model: {data.get('model')}")
                print(f"  Base URL: {data.get('base_url')}")
                
                model_info = data.get('model_info', {})
                if 'error' not in model_info:
                    print(f"  Model loaded successfully")
                
                print(f"\n✓ LLM status test passed")
                self.test_results.append(("LLM Status", True))
            else:
                print(f"✗ LLM service error: {data.get('error')}")
                self.test_results.append(("LLM Status", False))
                
        except Exception as e:
            print(f"✗ LLM status test failed: {str(e)}")
            self.test_results.append(("LLM Status", False))
    
    def test_new_conversation(self):
        """Test creating a new conversation"""
        print("\n" + "="*60)
        print("TEST 3: Create New Conversation")
        print("="*60)
        
        try:
            response = requests.post(f"{self.base_url}/conversation/new")
            data = response.json()
            
            if data.get('success'):
                self.conversation_id = data.get('conversation_id')
                print(f"✓ Created conversation: {self.conversation_id}")
                self.test_results.append(("Create Conversation", True))
            else:
                print(f"✗ Failed to create conversation: {data.get('error')}")
                self.test_results.append(("Create Conversation", False))
                
        except Exception as e:
            print(f"✗ Conversation creation failed: {str(e)}")
            self.test_results.append(("Create Conversation", False))
    
    def test_simple_chat(self):
        """Test basic chat functionality"""
        print("\n" + "="*60)
        print("TEST 4: Simple Chat Message")
        print("="*60)
        
        if not self.conversation_id:
            print("⚠  No conversation ID, creating new one...")
            self.test_new_conversation()
        
        message = "Hello! Can you help me explore my photos?"
        print(f"Sending: {message}")
        print("Waiting for response...")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    'message': message,
                    'conversation_id': self.conversation_id,
                    'top_k': 3
                },
                timeout=60
            )
            
            data = response.json()
            
            if data.get('success'):
                response_text = data.get('response', '')
                photos = data.get('retrieved_photos', [])
                validation = data.get('validation', {})
                
                print(f"\n✓ Received response ({len(response_text)} chars)")
                print(f"  Response preview: {response_text[:200]}...")
                print(f"  Retrieved photos: {len(photos)}")
                print(f"  Grounded: {validation.get('is_grounded', 'N/A')}")
                print(f"  Confidence: {validation.get('confidence', 'N/A')}")
                
                if response_text:
                    print(f"\n✓ Simple chat test passed")
                    self.test_results.append(("Simple Chat", True))
                else:
                    print(f"\n✗ Empty response")
                    self.test_results.append(("Simple Chat", False))
            else:
                print(f"✗ Chat failed: {data.get('error')}")
                self.test_results.append(("Simple Chat", False))
                
        except requests.exceptions.Timeout:
            print(f"✗ Chat request timed out (>60s)")
            self.test_results.append(("Simple Chat", False))
        except Exception as e:
            print(f"✗ Chat test failed: {str(e)}")
            self.test_results.append(("Simple Chat", False))
    
    def test_photo_query_chat(self):
        """Test chat with photo retrieval"""
        print("\n" + "="*60)
        print("TEST 5: Photo Query Chat")
        print("="*60)
        
        queries = [
            "Show me happy photos",
            "Find beach photos",
            "Photos from last summer"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        'message': query,
                        'conversation_id': self.conversation_id,
                        'top_k': 5
                    },
                    timeout=60
                )
                
                data = response.json()
                
                if data.get('success'):
                    photos = data.get('retrieved_photos', [])
                    response_text = data.get('response', '')
                    
                    print(f"  ✓ Retrieved {len(photos)} photos")
                    print(f"  Response: {response_text[:150]}...")
                else:
                    print(f"  ✗ Failed: {data.get('error')}")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print(f"\n✓ Photo query chat test complete")
        self.test_results.append(("Photo Query Chat", True))
    
    def test_conversation_continuity(self):
        """Test multi-turn conversation"""
        print("\n" + "="*60)
        print("TEST 6: Conversation Continuity")
        print("="*60)
        
        exchanges = [
            "Show me beach photos",
            "Who is in these photos?",
            "When were they taken?"
        ]
        
        print("Testing multi-turn conversation...")
        
        for i, message in enumerate(exchanges, 1):
            print(f"\nTurn {i}: {message}")
            
            try:
                response = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        'message': message,
                        'conversation_id': self.conversation_id,
                        'top_k': 3
                    },
                    timeout=60
                )
                
                data = response.json()
                
                if data.get('success'):
                    response_text = data.get('response', '')
                    print(f"  Response: {response_text[:120]}...")
                else:
                    print(f"  ✗ Failed: {data.get('error')}")
                    
                time.sleep(1)  # Brief pause between turns
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print(f"\n✓ Conversation continuity test complete")
        self.test_results.append(("Conversation Continuity", True))
    
    def test_insights_generation(self):
        """Test insights generation"""
        print("\n" + "="*60)
        print("TEST 7: Insights Generation")
        print("="*60)
        
        insight_types = ['emotional', 'social', 'general']
        
        for insight_type in insight_types:
            print(f"\nGenerating {insight_type} insights...")
            
            try:
                response = requests.post(
                    f"{self.base_url}/insights",
                    json={
                        'insight_type': insight_type
                    },
                    timeout=60
                )
                
                data = response.json()
                
                if data.get('success'):
                    insights = data.get('insights', '')
                    photos_analyzed = data.get('photos_analyzed', 0)
                    
                    print(f"  ✓ Analyzed {photos_analyzed} photos")
                    print(f"  Insights: {insights[:150]}...")
                else:
                    print(f"  ✗ Failed: {data.get('error')}")
                    
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        print(f"\n✓ Insights generation test complete")
        self.test_results.append(("Insights Generation", True))
    
    def test_streaming_chat(self):
        """Test streaming chat response"""
        print("\n" + "="*60)
        print("TEST 8: Streaming Chat")
        print("="*60)
        
        message = "Tell me about my photo collection"
        print(f"Sending: {message}")
        print("Receiving stream: ", end="", flush=True)
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/stream",
                json={
                    'message': message,
                    'conversation_id': self.conversation_id,
                    'top_k': 3
                },
                stream=True,
                timeout=60
            )
            
            chunks_received = 0
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            if data.get('type') == 'token':
                                print(".", end="", flush=True)
                                chunks_received += 1
                            elif data.get('type') == 'done':
                                print(f"\n\n  ✓ Received {chunks_received} chunks")
                                break
                        except:
                            pass
            
            if chunks_received > 0:
                print(f"✓ Streaming chat test passed")
                self.test_results.append(("Streaming Chat", True))
            else:
                print(f"✗ No chunks received")
                self.test_results.append(("Streaming Chat", False))
                
        except Exception as e:
            print(f"\n✗ Streaming test failed: {str(e)}")
            self.test_results.append(("Streaming Chat", False))
    
    def test_conversation_history(self):
        """Test retrieving conversation history"""
        print("\n" + "="*60)
        print("TEST 9: Conversation History")
        print("="*60)
        
        if not self.conversation_id:
            print("⚠  No conversation to retrieve")
            return
        
        try:
            response = requests.get(
                f"{self.base_url}/conversation/{self.conversation_id}"
            )
            
            data = response.json()
            
            if data.get('success'):
                messages = data.get('messages', [])
                stats = data.get('stats', {})
                
                print(f"✓ Retrieved conversation history")
                print(f"  Total messages: {len(messages)}")
                print(f"  User messages: {stats.get('user_messages', 0)}")
                print(f"  Assistant messages: {stats.get('assistant_messages', 0)}")
                print(f"  Photos discussed: {stats.get('total_photos_discussed', 0)}")
                
                print(f"\n✓ Conversation history test passed")
                self.test_results.append(("Conversation History", True))
            else:
                print(f"✗ Failed: {data.get('error')}")
                self.test_results.append(("Conversation History", False))
                
        except Exception as e:
            print(f"✗ History retrieval failed: {str(e)}")
            self.test_results.append(("Conversation History", False))
    
    def test_grounding_validation(self):
        """Test response grounding and validation"""
        print("\n" + "="*60)
        print("TEST 10: Response Grounding")
        print("="*60)
        
        # Ask a question that might cause hallucination
        message = "Did I go to Paris in 2019?"
        print(f"Testing with potentially misleading query: '{message}'")
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                json={
                    'message': message,
                    'conversation_id': self.conversation_id,
                    'top_k': 5
                },
                timeout=60
            )
            
            data = response.json()
            
            if data.get('success'):
                response_text = data.get('response', '').lower()
                validation = data.get('validation', {})
                
                print(f"\nResponse: {data.get('response')}")
                print(f"\nValidation:")
                print(f"  Grounded: {validation.get('is_grounded')}")
                print(f"  Confidence: {validation.get('confidence')}")
                print(f"  Warnings: {validation.get('warnings')}")
                
                # Check if response acknowledges uncertainty
                uncertainty_phrases = [
                    "don't have",
                    "not enough",
                    "can't determine",
                    "unclear",
                    "no information"
                ]
                
                acknowledges_uncertainty = any(
                    phrase in response_text 
                    for phrase in uncertainty_phrases
                )
                
                if acknowledges_uncertainty:
                    print(f"\n✓ LLM correctly acknowledged uncertainty")
                    print(f"✓ Grounding validation test passed")
                    self.test_results.append(("Grounding Validation", True))
                else:
                    print(f"\n⚠  LLM may have hallucinated (didn't acknowledge uncertainty)")
                    self.test_results.append(("Grounding Validation", False))
            else:
                print(f"✗ Failed: {data.get('error')}")
                self.test_results.append(("Grounding Validation", False))
                
        except Exception as e:
            print(f"✗ Grounding test failed: {str(e)}")
            self.test_results.append(("Grounding Validation", False))
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print(" "*20 + "PHASE 4 TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_ollama_connection,
            self.test_llm_status,
            self.test_new_conversation,
            self.test_simple_chat,
            self.test_photo_query_chat,
            self.test_conversation_continuity,
            self.test_insights_generation,
            self.test_streaming_chat,
            self.test_conversation_history,
            self.test_grounding_validation
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"\n✗ Test failed with error: {str(e)}")
                import traceback
                traceback.print_exc()
                self.test_results.append((test.__name__, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = len(self.test_results)
        passed = sum(1 for _, result in self.test_results if result)
        
        for test_name, result in self.test_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {test_name}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 All tests passed! Phase 4 is working correctly.")
            print("\nYour RAG system is complete and functional!")
        elif passed >= total * 0.7:
            print(f"\n⚠️  Most tests passed ({passed}/{total}). System is mostly functional.")
        else:
            print(f"\n❌ Many tests failed ({total - passed}/{total}). Please check logs.")


if __name__ == "__main__":
    print("Make sure:")
    print("1. Ollama is running: ollama serve")
    print("2. Model is downloaded: ollama pull llama3.2:3b")
    print("3. Backend is running: python app.py")
    print("")
    input("Press Enter to start tests...")
    
    tester = Phase4Tester()
    tester.run_all_tests()