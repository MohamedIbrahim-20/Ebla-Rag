#!/usr/bin/env python3
"""
Performance Testing Script for Ebla-RAG System
Tests response times, accuracy, and system performance with multiple datasets
"""

import os
import requests
import time
import json
import statistics
from typing import List, Dict, Tuple
import pandas as pd

class EblaRAGPerformanceTester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session_id = None
        self.results = []
        
    def create_session(self) -> str:
        """Create a new chat session by sending an empty message"""
        try:
            payload = {
                "message": "",
                "create_only": True,
                "method": "langchain"
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
                print(f"Created session: {self.session_id}")
                return self.session_id
            else:
                print(f"Failed to create session: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        except Exception as e:
            print(f"Error creating session: {e}")
            return None
    
    def index_dataset(self, project_id: str, csv_file: str) -> bool:
        """Index a dataset"""
        try:
            print(f"Indexing dataset: {csv_file}")
            start_time = time.time()
            
            # Extract just the filename from the path
            filename = os.path.basename(csv_file)
            print(filename)
            payload = {
                "csv_file": filename
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/data/index/{project_id}",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            indexing_time = end_time - start_time
            
            if response.status_code == 200:
                print(f"Successfully indexed {csv_file} in {indexing_time:.2f} seconds")
                return True
            else:
                print(f"Failed to index {csv_file}: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Error indexing dataset: {e}")
            return False
    
    def ask_question(self, question: str, method: str = "langchain") -> Tuple[Dict, float]:
        """Ask a question and measure response time"""
        try:
            start_time = time.time()
            
            payload = {
                "message": question,
                "session_id": self.session_id,
                "method": method
            }
            
            response = requests.post(
                f"{self.base_url}/api/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                return response.json(), response_time
            else:
                print(f"Failed to get response: {response.status_code}")
                return None, response_time
                
        except Exception as e:
            print(f"Error asking question: {e}")
            return None, 0
    
    def run_performance_test(self, test_questions: List[str], method: str = "langchain") -> Dict:
        """Run performance tests with a list of questions"""
        print(f"\n=== Running Performance Test with {method.upper()} ===")
        
        response_times = []
        successful_queries = 0
        failed_queries = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\nQuestion {i}/{len(test_questions)}: {question[:50]}...")
            
            result, response_time = self.ask_question(question, method)
            response_times.append(response_time)
            
            if result:
                successful_queries += 1
                print(f"✓ Response time: {response_time:.2f}s")
                
                # Store detailed results
                self.results.append({
                    "question": question,
                    "method": method,
                    "response_time": response_time,
                    "success": True,
                    "answer_length": len(result.get("answer", "")),
                    "sources_count": len(result.get("sources", []))
                })
            else:
                failed_queries += 1
                print(f"✗ Failed - Response time: {response_time:.2f}s")
                
                self.results.append({
                    "question": question,
                    "method": method,
                    "response_time": response_time,
                    "success": False,
                    "answer_length": 0,
                    "sources_count": 0
                })
        
        # Calculate statistics
        stats = {
            "method": method,
            "total_questions": len(test_questions),
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": (successful_queries / len(test_questions)) * 100,
            "avg_response_time": statistics.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "median_response_time": statistics.median(response_times)
        }
        
        return stats
    
    def print_performance_report(self, stats: Dict):
        """Print a formatted performance report"""
        print(f"\n{'='*60}")
        print(f"PERFORMANCE REPORT - {stats['method'].upper()}")
        print(f"{'='*60}")
        print(f"Total Questions: {stats['total_questions']}")
        print(f"Successful Queries: {stats['successful_queries']}")
        print(f"Failed Queries: {stats['failed_queries']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"\nResponse Time Statistics:")
        print(f"  Average: {stats['avg_response_time']:.2f}s")
        print(f"  Minimum: {stats['min_response_time']:.2f}s")
        print(f"  Maximum: {stats['max_response_time']:.2f}s")
        print(f"  Median: {stats['median_response_time']:.2f}s")
    
    def save_results_to_csv(self, filename: str = "performance_results.csv"):
        """Save detailed results to CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"\nDetailed results saved to: {filename}")

def main():
    """Main testing function"""
    tester = EblaRAGPerformanceTester()
    
    # Test questions for different domains
    tech_questions = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are the differences between deep learning and traditional machine learning?",
        "Explain natural language processing and its applications",
        "What is computer vision and how is it used?",
        "How does blockchain technology work?",
        "What are the benefits of cloud computing?",
        "Explain cybersecurity best practices"
    ]
    
    science_questions = [
        "How does photosynthesis work?",
        "What is the structure of DNA?",
        "Explain the theory of evolution",
        "What is quantum mechanics?",
        "How does Einstein's theory of relativity work?",
        "What is cell biology?",
        "Explain genetics and heredity",
        "What is ecology and why is it important?"
    ]
    
    cross_domain_questions = [
        "How does quantum computing relate to quantum mechanics?",
        "What are the applications of AI in biology?",
        "How can machine learning be used in climate science?",
        "What is the relationship between computer vision and neuroscience?"
    ]
    
    print("Starting Ebla-RAG Performance Testing...")
    
    # Create session
    if not tester.create_session():
        print("Failed to create session. Exiting.")
        return
    
    # Index datasets
    print("\n=== Indexing Datasets ===")
    # tech_indexed = tester.index_dataset("2", "tech_100_long_real.csv")
    # tech_indexed = tester.index_dataset("tech", "tech_100_long_real.csv")
    tech_indexed = tester.index_dataset("2", "tech_dataset.csv")
    science_indexed = tester.index_dataset("2", "science_dataset.csv")
    
    if not (tech_indexed and science_indexed):
        print("Failed to index datasets. Exiting.")
        return
    
    # Wait for indexing to complete
    print("Waiting 5 seconds for indexing to complete...")
    time.sleep(5)
    
    # Run performance tests
    all_stats = []
    
    # Test with technology questions
    tech_stats = tester.run_performance_test(tech_questions, "langchain")
    tester.print_performance_report(tech_stats)
    all_stats.append(tech_stats)
    
    # Test with science questions
    science_stats = tester.run_performance_test(science_questions, "langchain")
    tester.print_performance_report(science_stats)
    all_stats.append(science_stats)
    
    # Test with cross-domain questions
    cross_stats = tester.run_performance_test(cross_domain_questions, "langchain")
    tester.print_performance_report(cross_stats)
    all_stats.append(cross_stats)
    
    # Overall summary
    total_questions = sum(s['total_questions'] for s in all_stats)
    total_successful = sum(s['successful_queries'] for s in all_stats)
    avg_response_times = [s['avg_response_time'] for s in all_stats]
    
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Questions Tested: {total_questions}")
    print(f"Overall Success Rate: {(total_successful/total_questions)*100:.1f}%")
    print(f"Average Response Time: {statistics.mean(avg_response_times):.2f}s")
    
    # Save results
    tester.save_results_to_csv()
    
    print("\nPerformance testing completed!")

if __name__ == "__main__":
    main()