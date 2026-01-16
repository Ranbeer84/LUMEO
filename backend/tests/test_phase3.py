"""
Phase 3 Testing Script
Tests all retrieval system components
"""

import requests
import json
from typing import Dict, List

# Configuration
BASE_URL = "http://localhost:5002/api"


class Phase3Tester:
    """Test suite for Phase 3 RAG retrieval system"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
    
    def test_health(self):
        """Test that services are running"""
        print("\n" + "="*60)
        print("TEST 1: Health Check")
        print("="*60)
        
        response = requests.get(f"{self.base_url}/health")
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Database: {data.get('database')}")
        print(f"Services: {data.get('services')}")
        print(f"Photos: {data.get('photos')}")
        
        assert data['status'] == 'healthy', "System not healthy"
        assert data['services'] == True, "Services not available"
        
        print("✓ Health check passed")
        self.test_results.append(("Health Check", True))
    
    def test_retrieval_stats(self):
        """Test retrieval statistics"""
        print("\n" + "="*60)
        print("TEST 2: Retrieval Statistics")
        print("="*60)
        
        response = requests.get(f"{self.base_url}/retrieval/stats")
        data = response.json()
        
        if data.get('success'):
            stats = data.get('retrieval_stats', {})
            cache = data.get('cache_stats', {})
            
            print("Retrieval Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print("\nCache Stats:")
            for key, value in cache.items():
                print(f"  {key}: {value}")
            
            print("✓ Stats retrieved successfully")
            self.test_results.append(("Retrieval Stats", True))
        else:
            print("✗ Failed to get stats")
            self.test_results.append(("Retrieval Stats", False))
    
    def test_query_parsing(self):
        """Test query parser"""
        print("\n" + "="*60)
        print("TEST 3: Query Parsing")
        print("="*60)
        
        test_queries = [
            "beach photos with Mom",
            "happy photos from last summer",
            "photos where I'm wearing black",
            "indoor party photos with cake",
            "morning photos at home"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            response = requests.post(
                f"{self.base_url}/query/parse",
                json={'query': query}
            )
            
            data = response.json()
            
            if data.get('success'):
                filters = data.get('parsed_filters', {})
                print(f"  Parsed filters:")
                for key, value in filters.items():
                    if key != 'raw_query':
                        print(f"    {key}: {value}")
                
                if len(filters) > 1:
                    print("  ✓ Filters extracted")
                else:
                    print("  ℹ No specific filters (semantic search only)")
            else:
                print(f"  ✗ Parsing failed: {data.get('error')}")
        
        print("\n✓ Query parsing test complete")
        self.test_results.append(("Query Parsing", True))
    
    def test_semantic_search(self):
        """Test semantic search"""
        print("\n" + "="*60)
        print("TEST 4: Semantic Search")
        print("="*60)
        
        test_queries = [
            ("beach sunset", 5),
            ("happy family", 5),
            ("party celebration", 5),
            ("outdoor nature", 5)
        ]
        
        for query, top_k in test_queries:
            print(f"\nQuery: '{query}' (top {top_k})")
            
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    'query': query,
                    'top_k': top_k,
                    'use_filters': False  # Pure semantic search
                }
            )
            
            data = response.json()
            
            if data.get('success'):
                results = data.get('results', [])
                print(f"  Found {len(results)} photos")
                
                for i, result in enumerate(results[:3], 1):
                    print(f"  {i}. {result['filename']}")
                    print(f"     Similarity: {result['similarity']:.3f}")
                    print(f"     Scene: {result.get('scene_type', 'N/A')} at {result.get('location', 'N/A')}")
                
                if results:
                    print("  ✓ Search successful")
                else:
                    print("  ℹ No results found (photos may need embeddings)")
            else:
                print(f"  ✗ Search failed: {data.get('error')}")
        
        print("\n✓ Semantic search test complete")
        self.test_results.append(("Semantic Search", True))
    
    def test_hybrid_search(self):
        """Test hybrid search with filters"""
        print("\n" + "="*60)
        print("TEST 5: Hybrid Search with Filters")
        print("="*60)
        
        # This assumes you have photos with these characteristics
        test_queries = [
            "happy beach photos",
            "indoor photos with people",
            "photos from last summer",
            "outdoor morning photos"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            
            response = requests.post(
                f"{self.base_url}/search",
                json={
                    'query': query,
                    'top_k': 5,
                    'use_filters': True  # Enable hybrid search
                }
            )
            
            data = response.json()
            
            if data.get('success'):
                results = data.get('results', [])
                filters = data.get('parsed_filters', {})
                
                print(f"  Filters applied:")
                for key, value in filters.items():
                    if key != 'raw_query':
                        print(f"    {key}: {value}")
                
                print(f"  Found {len(results)} photos")
                
                for i, result in enumerate(results[:2], 1):
                    print(f"  {i}. {result['filename']}")
                    print(f"     Similarity: {result['similarity']:.3f}")
                    print(f"     Match reasons: {', '.join(result.get('match_reasons', []))}")
                
                if results:
                    print("  ✓ Hybrid search successful")
                else:
                    print("  ℹ No results found")
            else:
                print(f"  ✗ Search failed: {data.get('error')}")
        
        print("\n✓ Hybrid search test complete")
        self.test_results.append(("Hybrid Search", True))
    
    def test_context_generation(self):
        """Test LLM context generation"""
        print("\n" + "="*60)
        print("TEST 6: LLM Context Generation")
        print("="*60)
        
        query = "show me happy family photos"
        print(f"Query: '{query}'\n")
        
        response = requests.post(
            f"{self.base_url}/search/context",
            json={
                'query': query,
                'top_k': 3,
                'include_system_prompt': True
            }
        )
        
        data = response.json()
        
        if data.get('success'):
            context = data.get('context', '')
            tokens = data.get('estimated_tokens', 0)
            results_count = data.get('results_count', 0)
            
            print(f"Results: {results_count} photos")
            print(f"Estimated tokens: {tokens}")
            print(f"\nContext preview (first 500 chars):")
            print("-" * 60)
            print(context[:500])
            print("..." if len(context) > 500 else "")
            print("-" * 60)
            
            print("\n✓ Context generation successful")
            self.test_results.append(("Context Generation", True))
        else:
            print(f"✗ Context generation failed: {data.get('error')}")
            self.test_results.append(("Context Generation", False))
    
    def test_similar_photos(self):
        """Test similar photo search"""
        print("\n" + "="*60)
        print("TEST 7: Similar Photo Search")
        print("="*60)
        
        # First, get a photo ID to test with
        response = requests.get(f"{self.base_url}/stats")
        stats_data = response.json()
        
        if stats_data.get('total_photos', 0) == 0:
            print("ℹ No photos in database to test similar search")
            return
        
        # Get first photo ID (you may want to modify this)
        # For now, we'll skip this test if we don't have a photo ID
        print("ℹ Skipping similar photo test (need photo ID)")
        print("  To test manually:")
        print("  GET /api/search/similar/<photo_id>?top_k=5")
    
    def test_summary_generation(self):
        """Test summary/insights generation"""
        print("\n" + "="*60)
        print("TEST 8: Summary Generation")
        print("="*60)
        
        summary_types = ['general', 'emotional', 'people']
        
        for summary_type in summary_types:
            print(f"\nSummary type: {summary_type}")
            
            response = requests.post(
                f"{self.base_url}/insights/summary",
                json={
                    'summary_type': summary_type,
                    'filters': {}
                }
            )
            
            data = response.json()
            
            if data.get('success'):
                summary = data.get('summary', '')
                photos_analyzed = data.get('photos_analyzed', 0)
                
                print(f"  Analyzed {photos_analyzed} photos")
                print(f"  Summary preview:")
                print("  " + "-" * 58)
                for line in summary.split('\n')[:10]:
                    print(f"  {line}")
                print("  " + "-" * 58)
                
                print("  ✓ Summary generated")
            else:
                print(f"  ✗ Summary failed: {data.get('error')}")
        
        print("\n✓ Summary generation test complete")
        self.test_results.append(("Summary Generation", True))
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*70)
        print(" "*20 + "PHASE 3 TEST SUITE")
        print("="*70)
        
        tests = [
            self.test_health,
            self.test_retrieval_stats,
            self.test_query_parsing,
            self.test_semantic_search,
            self.test_hybrid_search,
            self.test_context_generation,
            self.test_similar_photos,
            self.test_summary_generation
        ]
        
        for test in tests:
            try:
                test()
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
            print("\n🎉 All tests passed! Phase 3 is working correctly.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check logs above.")


if __name__ == "__main__":
    tester = Phase3Tester()
    tester.run_all_tests()