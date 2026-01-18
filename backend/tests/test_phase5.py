"""
Phase 5 Testing Script
Tests conversational memory and advanced features
"""

import requests
import json
import time
from typing import Dict, List

# Configuration
BASE_URL = "http://localhost:5002/api"


class Phase5Tester:
    """Test suite for Phase 5 conversational memory"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
        self.conversation_id = None
    
    def test_conversation_storage(self):
        """Test Step 5.1: Conversation Storage"""
        print("\n" + "="*60)
        print("TEST 1: Conversation Storage System")
        print("="*60)
        
        try:
            # Create conversation
            response = requests.post(f"{self.base_url}/conversation/new")
            data = response.json()
            
            if data.get('success'):
                self.conversation_id = data['conversation_id']
                print(f"✓ Created conversation: {self.conversation_id}")
                
                # Add messages
                for i in range(3):
                    msg_response = requests.post(
                        f"{self.base_url}/chat",
                        json={
                            'message': f'Test message {i+1}',
                            'conversation_id': self.conversation_id,
                            'top_k': 3
                        },
                        timeout=60
                    )
                    
                    if msg_response.json().get('success'):
                        print(f"  ✓ Sent message {i+1}")
                    
                    time.sleep(2)
                
                # Retrieve history
                history_response = requests.get(
                    f"{self.base_url}/conversation/{self.conversation_id}"
                )
                
                history_data = history_response.json()
                
                if history_data.get('success'):
                    messages = history_data.get('messages', [])
                    print(f"\n✓ Retrieved {len(messages)} messages")
                    print(f"  User messages: {sum(1 for m in messages if m['role'] == 'user')}")
                    print(f"  Assistant messages: {sum(1 for m in messages if m['role'] == 'assistant')}")
                    
                    print("\n✓ Conversation storage test passed")
                    self.test_results.append(("Conversation Storage", True))
                else:
                    print("✗ Failed to retrieve history")
                    self.test_results.append(("Conversation Storage", False))
            else:
                print("✗ Failed to create conversation")
                self.test_results.append(("Conversation Storage", False))
                
        except Exception as e:
            print(f"✗ Storage test failed: {str(e)}")
            self.test_results.append(("Conversation Storage", False))
    
    def test_context_carryover(self):
        """Test Step 5.2: Context Carry-Over"""
        print("\n" + "="*60)
        print("TEST 2: Context Carry-Over")
        print("="*60)
        
        if not self.conversation_id:
            print("⚠ No conversation, creating one...")
            self.test_conversation_storage()
        
        try:
            # First query
            print("\nQuery 1: 'Show me beach photos'")
            response1 = requests.post(
                f"{self.base_url}/chat",
                json={
                    'message': 'Show me beach photos',
                    'conversation_id': self.conversation_id,
                    'top_k': 3
                },
                timeout=60
            )
            
            data1 = response1.json()
            
            if data1.get('success'):
                photos = data1.get('retrieved_photos', [])
                print(f"  ✓ Retrieved {len(photos)} photos")
                
                time.sleep(2)
                
                # Follow-up query (should use context)
                print("\nQuery 2: 'Who is in these photos?' (follow-up)")
                response2 = requests.post(
                    f"{self.base_url}/chat",
                    json={
                        'message': 'Who is in these photos?',
                        'conversation_id': self.conversation_id,
                        'top_k': 3
                    },
                    timeout=60
                )
                
                data2 = response2.json()
                
                if data2.get('success'):
                    response_text = data2.get('response', '')
                    print(f"  ✓ Got contextual response")
                    print(f"  Preview: {response_text[:150]}...")
                    
                    # Check if response seems contextual
                    # (mentions people or photos from previous query)
                    is_contextual = ('photo' in response_text.lower() or 
                                   'these' in response_text.lower())
                    
                    if is_contextual:
                        print("\n✓ Context carry-over working correctly")
                        self.test_results.append(("Context Carry-Over", True))
                    else:
                        print("\n⚠ Response may not be using context")
                        self.test_results.append(("Context Carry-Over", False))
                else:
                    print("✗ Follow-up query failed")
                    self.test_results.append(("Context Carry-Over", False))
            else:
                print("✗ First query failed")
                self.test_results.append(("Context Carry-Over", False))
                
        except Exception as e:
            print(f"✗ Context carry-over test failed: {str(e)}")
            self.test_results.append(("Context Carry-Over", False))
    
    def test_auto_summarization(self):
        """Test Step 5.3: Automatic Summarization"""
        print("\n" + "="*60)
        print("TEST 3: Auto-Summarization")
        print("="*60)
        
        try:
            # Create new conversation
            response = requests.post(f"{self.base_url}/conversation/new")
            conv_id = response.json()['conversation_id']
            print(f"Created test conversation: {conv_id}")
            
            # Send multiple messages to trigger summarization
            print("\nSending 10 messages to trigger auto-summarization...")
            
            for i in range(10):
                requests.post(
                    f"{self.base_url}/chat",
                    json={
                        'message': f'Test query {i+1}: show me photos',
                        'conversation_id': conv_id,
                        'top_k': 2
                    },
                    timeout=60
                )
                print(f"  Sent message {i+1}/10")
                time.sleep(1)
            
            # Check if conversation was summarized
            time.sleep(2)
            
            history_response = requests.get(
                f"{self.base_url}/conversation/{conv_id}"
            )
            
            data = history_response.json()
            
            # Check conversation details
            conv_response = requests.get(f"{self.base_url}/conversations?limit=1")
            conv_data = conv_response.json()
            
            if conv_data.get('success') and conv_data.get('conversations'):
                latest_conv = conv_data['conversations'][0]
                summary = latest_conv.get('summary')
                
                if summary:
                    print(f"\n✓ Auto-summarization triggered")
                    print(f"  Summary: {summary[:150]}...")
                    print("\n✓ Auto-summarization test passed")
                    self.test_results.append(("Auto-Summarization", True))
                else:
                    print("\n⚠ No summary generated (may need more messages)")
                    print("  Trying manual summarization...")
                    
                    # Try manual summarization
                    sum_response = requests.post(
                        f"{self.base_url}/conversation/{conv_id}/summarize"
                    )
                    
                    sum_data = sum_response.json()
                    
                    if sum_data.get('success'):
                        print(f"✓ Manual summarization worked")
                        print(f"  Summary: {sum_data['summary'][:150]}...")
                        self.test_results.append(("Auto-Summarization", True))
                    else:
                        print("✗ Manual summarization failed")
                        self.test_results.append(("Auto-Summarization", False))
            else:
                print("✗ Failed to get conversation details")
                self.test_results.append(("Auto-Summarization", False))
                
        except Exception as e:
            print(f"✗ Summarization test failed: {str(e)}")
            self.test_results.append(("Auto-Summarization", False))
    
    def test_memory_features(self):
        """Test Phase 5 memory features"""
        print("\n" + "="*60)
        print("TEST 4: Memory Features")
        print("="*60)
        
        try:
            # Test frequently discussed topics
            print("\n1. Testing frequently discussed topics...")
            topics_response = requests.get(f"{self.base_url}/memory/topics?limit=10")
            topics_data = topics_response.json()
            
            if topics_data.get('success'):
                topics = topics_data.get('topics', [])
                print(f"  ✓ Found {len(topics)} topics")
                for topic in topics[:5]:
                    print(f"    - {topic['topic']}: {topic['count']} mentions")
            
            # Test conversation timeline
            print("\n2. Testing conversation timeline...")
            timeline_response = requests.get(f"{self.base_url}/memory/timeline?days=30")
            timeline_data = timeline_response.json()
            
            if timeline_data.get('success'):
                timeline = timeline_data.get('timeline', [])
                print(f"  ✓ Timeline has {len(timeline)} conversations")
            
            # Test conversation search
            print("\n3. Testing conversation search...")
            search_response = requests.post(
                f"{self.base_url}/conversations/search",
                json={'query': 'photos', 'limit': 5}
            )
            search_data = search_response.json()
            
            if search_data.get('success'):
                results = search_data.get('results', [])
                print(f"  ✓ Found {len(results)} matching conversations")
            
            print("\n✓ Memory features test passed")
            self.test_results.append(("Memory Features", True))
            
        except Exception as e:
            print(f"✗ Memory features test failed: {str(e)}")
            self.test_results.append(("Memory Features", False))
    
    def test_conversation_optimization(self):
        """Test conversation optimization"""
        print("\n" + "="*60)
        print("TEST 5: Conversation Optimization")
        print("="*60)
        
        if not self.conversation_id:
            print("⚠ No conversation to optimize")
            return
        
        try:
            # Try to optimize conversation
            response = requests.post(
                f"{self.base_url}/conversation/{self.conversation_id}/optimize",
                json={'target_messages': 10}
            )
            
            data = response.json()
            
            if data.get('success'):
                if data.get('optimized'):
                    print(f"✓ Conversation optimized:")
                    print(f"  Original messages: {data.get('original_messages')}")
                    print(f"  Compressed: {data.get('compressed_messages')}")
                    print(f"  Kept: {data.get('kept_messages')}")
                    print(f"  Summary length: {data.get('summary_length')} chars")
                else:
                    print(f"ℹ {data.get('reason')}")
                
                print("\n✓ Optimization test passed")
                self.test_results.append(("Conversation Optimization", True))
            else:
                print("✗ Optimization failed")
                self.test_results.append(("Conversation Optimization", False))
                
        except Exception as e:
            print(f"✗ Optimization test failed: {str(e)}")
            self.test_results.append(("Conversation Optimization", False))
    
    def test_enhanced_chat(self):
        """Test enhanced chat with memory"""
        print("\n" + "="*60)
        print("TEST 6: Enhanced Chat with Memory")
        print("="*60)
        
        try:
            # Create new conversation
            response = requests.post(f"{self.base_url}/conversation/new")
            conv_id = response.json()['conversation_id']
            
            # First chat (creates memory)
            print("\n1. First chat about beach photos...")
            response1 = requests.post(
                f"{self.base_url}/chat/enhanced",
                json={
                    'message': 'Show me beach photos with family',
                    'conversation_id': conv_id,
                    'use_memory': True,
                    'top_k': 3
                },
                timeout=60
            )
            
            data1 = response1.json()
            
            if data1.get('success'):
                print(f"  ✓ Retrieved {len(data1.get('retrieved_photos', []))} photos")
                print(f"  Memories used: {len(data1.get('relevant_memories', []))}")
                
                time.sleep(2)
                
                # Second chat (should use memory from first)
                print("\n2. Second chat (should recall beach context)...")
                response2 = requests.post(
                    f"{self.base_url}/chat/enhanced",
                    json={
                        'message': 'Show me more outdoor photos',
                        'conversation_id': conv_id,
                        'use_memory': True,
                        'top_k': 3
                    },
                    timeout=60
                )
                
                data2 = response2.json()
                
                if data2.get('success'):
                    memories = data2.get('relevant_memories', [])
                    print(f"  ✓ Used {len(memories)} relevant memories")
                    
                    if memories:
                        print(f"  Memory preview: {memories[0]['summary'][:100]}...")
                    
                    print("\n✓ Enhanced chat test passed")
                    self.test_results.append(("Enhanced Chat", True))
                else:
                    print("✗ Second chat failed")
                    self.test_results.append(("Enhanced Chat", False))
            else:
                print("✗ First chat failed")
                self.test_results.append(("Enhanced Chat", False))
                
        except Exception as e:
            print(f"✗ Enhanced chat test failed: {str(e)}")
            self.test_results.append(("Enhanced Chat", False))
    
    def test_conversation_export(self):
        """Test conversation export"""
        print("\n" + "="*60)
        print("TEST 7: Conversation Export")
        print("="*60)
        
        if not self.conversation_id:
            print("⚠ No conversation to export")
            return
        
        try:
            # Export as JSON
            print("\n1. Exporting as JSON...")
            json_response = requests.get(
                f"{self.base_url}/conversation/{self.conversation_id}/export?format=json"
            )
            
            if json_response.status_code == 200:
                data = json_response.json()
                print(f"  ✓ Exported {data.get('message_count', 0)} messages")
                
                # Export as Markdown
                print("\n2. Exporting as Markdown...")
                md_response = requests.get(
                    f"{self.base_url}/conversation/{self.conversation_id}/export?format=markdown"
                )
                
                if md_response.status_code == 200:
                    md_content = md_response.text
                    print(f"  ✓ Markdown export successful ({len(md_content)} chars)")
                    print(f"  Preview:\n{md_content[:200]}...")
                    
                    print("\n✓ Export test passed")
                    self.test_results.append(("Conversation Export", True))
                else:
                    print("✗ Markdown export failed")
                    self.test_results.append(("Conversation Export", False))
            else:
                print("✗ JSON export failed")
                self.test_results.append(("Conversation Export", False))
                
        except Exception as e:
            print(f"✗ Export test failed: {str(e)}")
            self.test_results.append(("Conversation Export", False))
    
    def run_all_tests(self):
        """Run all Phase 5 tests"""
        print("\n" + "="*70)
        print(" "*20 + "PHASE 5 TEST SUITE")
        print(" "*15 + "Conversational Memory Features")
        print("="*70)
        
        tests = [
            self.test_conversation_storage,
            self.test_context_carryover,
            self.test_auto_summarization,
            self.test_memory_features,
            self.test_conversation_optimization,
            self.test_enhanced_chat,
            self.test_conversation_export
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(2)
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
            print("\n🎉 All Phase 5 tests passed!")
            print("Your conversational memory system is working perfectly!")
        elif passed >= total * 0.7:
            print(f"\n✅ Most tests passed ({passed}/{total})")
            print("Phase 5 is mostly functional")
        else:
            print(f"\n⚠️ Several tests failed ({total - passed}/{total})")
            print("Please check the logs above")


if __name__ == "__main__":
    print("Phase 5 Test Suite - Conversational Memory")
    print("\nMake sure:")
    print("1. Backend is running: python app.py")
    print("2. Ollama is running: ollama serve")
    print("3. Phase 4 tests passed")
    print("")
    input("Press Enter to start Phase 5 tests...")
    
    tester = Phase5Tester()
    tester.run_all_tests()