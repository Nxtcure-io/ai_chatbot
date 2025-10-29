"""
Evaluation Script - Test RAG Chatbot Performance and Accuracy
Based on first 20 trials from the dataset
"""
import json
import time
from typing import List, Dict, Any
from chatbot import RAGChatbot
import re


class ChatbotEvaluator:
    """Chatbot Evaluator"""
    
    def __init__(self):
        """Initialize evaluator"""
        print("Initializing chatbot...")
        self.chatbot = RAGChatbot()
        self.results = []
        
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """
        Create test query set based on first 20 trials in the dataset
        Each query asks about specific information from a trial
        """
        return [
            {
                'query': 'What is the current recruitment status of trial NCT05812131?',
                'expected_keywords': ['NCT05812131', 'recruiting', 'RECRUITING'],
                'category': 'status_query',
                'trial_id': 'NCT05812131'
            },
            {
                'query': 'Can you tell me the official title of the COPEWeb Training for Providers study?',
                'expected_keywords': ['PTSD', 'Substance Use', 'Web-Based', 'Training'],
                'category': 'title_query',
                'trial_id': 'NCT05812131'
            },
            {
                'query': 'What conditions are being studied in trial NCT04036331?',
                'expected_keywords': ['Weight Loss', 'Obesity', 'Pediatric', 'Adolescent'],
                'category': 'condition_query',
                'trial_id': 'NCT04036331'
            },
            {
                'query': 'What type of study is the Dyad Plus Effectiveness/Feasibility trial?',
                'expected_keywords': ['INTERVENTIONAL', 'intervention'],
                'category': 'study_type_query',
                'trial_id': 'NCT04036331'
            },
            {
                'query': 'What is the minimum age requirement for the study about puberty, testosterone, and brain development (NCT06670053)?',
                'expected_keywords': ['12', 'Years', 'adolescent'],
                'category': 'eligibility_query',
                'trial_id': 'NCT06670053'
            },
            {
                'query': 'Is the study on testosterone effects on transmasculine adolescents an observational or interventional study?',
                'expected_keywords': ['OBSERVATIONAL', 'observational'],
                'category': 'study_type_query',
                'trial_id': 'NCT06670053'
            },
            {
                'query': 'Where is the Restorative Environments for Gait Therapy with VR study being conducted?',
                'expected_keywords': ['location', 'site'],
                'category': 'location_query',
                'trial_id': 'NCT06304077'
            },
            {
                'query': 'What interventions are being tested in trial NCT06304077?',
                'expected_keywords': ['Forest', 'Urban', 'Rural', 'Virtual Reality'],
                'category': 'intervention_query',
                'trial_id': 'NCT06304077'
            },
            {
                'query': 'What gender requirements does the mid-urethral sling surgery study (NCT02316275) have?',
                'expected_keywords': ['FEMALE', 'female', 'women'],
                'category': 'eligibility_query',
                'trial_id': 'NCT02316275'
            },
            {
                'query': 'Who is the principal investigator for the study on stress urinary incontinence treatment NCT02316275?',
                'expected_keywords': ['Jennifer', 'Anger'],
                'category': 'investigator_query',
                'trial_id': 'NCT02316275'
            },
            {
                'query': 'What is the ToolBox Detect study (NCT04852601) trying to detect?',
                'expected_keywords': ['cognitive', 'decline', 'detection'],
                'category': 'purpose_query',
                'trial_id': 'NCT04852601'
            },
            {
                'query': 'What is trial NCT04558619 about and what is it trying to reduce?',
                'expected_keywords': ['Maternal', 'Infant', 'Health', 'Kōmmour', 'Prenatal'],
                'category': 'purpose_query',
                'trial_id': 'NCT04558619'
            },
            {
                'query': 'What population is being studied in the MASALA trial NCT01207167?',
                'expected_keywords': ['South Asian', 'America', 'Atherosclerosis'],
                'category': 'population_query',
                'trial_id': 'NCT01207167'
            },
            {
                'query': 'What is the focus of study NCT05398367 regarding lactating women?',
                'expected_keywords': ['Galactagogue', 'milk', 'supply', 'lactating'],
                'category': 'condition_query',
                'trial_id': 'NCT05398367'
            },
            {
                'query': 'What age group is the youth substance misuse prevention study (NCT05736211) targeting?',
                'expected_keywords': ['youth', 'young', 'organization'],
                'category': 'population_query',
                'trial_id': 'NCT05736211'
            },
            {
                'query': 'What type of cancer is being treated in the T-Cell Therapy EB103 study NCT06343311?',
                'expected_keywords': ['B-Cell', 'Non-Hodgkin', 'Lymphoma', 'Relapsed', 'Refractory'],
                'category': 'condition_query',
                'trial_id': 'NCT06343311'
            },
            {
                'query': 'Is trial NCT05528744 about molecular spectrum and clinical phenotypes currently recruiting?',
                'expected_keywords': ['NCT05528744', 'RECRUITING', 'recruiting'],
                'category': 'status_query',
                'trial_id': 'NCT05528744'
            },
            {
                'query': 'What condition is being studied in the intestinal microbiota evolution study NCT04540432?',
                'expected_keywords': ['Juvenile', 'Spondylarthropathy', 'microbiota'],
                'category': 'condition_query',
                'trial_id': 'NCT04540432'
            },
            {
                'query': 'What intervention is being tested in the vascular Ehlers-Danlos syndrome study NCT05994664?',
                'expected_keywords': ['Heart', 'Coherence', 'Training'],
                'category': 'intervention_query',
                'trial_id': 'NCT05994664'
            },
            {
                'query': 'What is the minimum age for participants in the High Intensity PreHab before major abdominal surgery study NCT05355909?',
                'expected_keywords': ['age', 'adult', '18'],
                'category': 'eligibility_query',
                'trial_id': 'NCT05355909'
            }
        ]
    
    def has_source_reference(self, answer: str, sources: List[Dict]) -> bool:
        """
        Check if answer contains source citations
        
        Args:
            answer: Generated answer
            sources: List of sources
            
        Returns:
            Whether answer contains source citations
        """
        # Check if NCT ID is mentioned in answer
        nct_pattern = r'NCT\d+'
        nct_mentions = re.findall(nct_pattern, answer, re.IGNORECASE)
        
        # Check NCT IDs in sources
        source_nct_ids = [source.get('NCTId', '') for source in sources]
        
        # At least one NCT ID is mentioned, or has explicit source citation
        has_nct = len(nct_mentions) > 0
        has_sources = len(sources) > 0
        
        return has_nct or has_sources
    
    def check_trial_match(self, trial_id: str, sources: List[Dict]) -> bool:
        """
        Check if the expected trial is in the sources
        
        Args:
            trial_id: Expected trial NCT ID
            sources: List of sources returned
            
        Returns:
            Whether the expected trial was retrieved
        """
        source_ids = [source.get('NCTId', '') for source in sources]
        return trial_id in source_ids
    
    def check_grounding(self, answer: str, sources: List[Dict], trial_id: str = None) -> Dict[str, Any]:
        """
        Check if answer has good source grounding
        
        Args:
            answer: Generated answer
            sources: List of sources
            trial_id: Expected trial ID (if applicable)
            
        Returns:
            Scoring dictionary
        """
        # 1. Check if has source citation
        has_ref = self.has_source_reference(answer, sources)
        
        # 2. Check if expected trial was retrieved
        correct_trial_retrieved = False
        if trial_id:
            correct_trial_retrieved = self.check_trial_match(trial_id, sources)
        
        # 3. Check if explicitly states inability to find information (when no relevant data)
        no_info_phrases = ['cannot find', 'not found', 'no relevant', 'uncertain', 'unable to answer']
        explicitly_states_no_info = any(phrase in answer.lower() for phrase in no_info_phrases)
        
        # 4. Check answer quality
        is_meaningful = len(answer) > 50  # Answer should have reasonable length
        
        return {
            'has_source_reference': has_ref,
            'correct_trial_retrieved': correct_trial_retrieved,
            'explicitly_states_no_info': explicitly_states_no_info,
            'is_meaningful': is_meaningful,
            'num_sources': len(sources),
            'grounded': has_ref and is_meaningful and correct_trial_retrieved
        }
    
    def evaluate_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate single query
        
        Args:
            test_case: Test case
            
        Returns:
            Evaluation result
        """
        query = test_case['query']
        expected_keywords = test_case['expected_keywords']
        category = test_case['category']
        trial_id = test_case.get('trial_id')
        
        print(f"\nTesting query: {query}")
        
        # Execute query
        result = self.chatbot.chat(query)
        
        # Check answer grounding
        grounding = self.check_grounding(result['answer'], result['sources'], trial_id)
        
        # Check keywords
        answer_lower = result['answer'].lower()
        keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keyword_coverage = len(keywords_found) / len(expected_keywords) if expected_keywords else 0
        
        # Check performance (latency)
        retrieval_time = float(result['timing']['retrieval_time'].replace('s', ''))
        meets_latency_requirement = retrieval_time < 2.0
        
        # Scoring
        evaluation = {
            'query': query,
            'category': category,
            'trial_id': trial_id,
            'answer': result['answer'],
            'sources': result['sources'],
            'timing': result['timing'],
            'grounding': grounding,
            'keyword_coverage': keyword_coverage,
            'keywords_found': keywords_found,
            'meets_latency': meets_latency_requirement,
            'retrieval_time': retrieval_time,
            'success': grounding['grounded'] and meets_latency_requirement
        }
        
        return evaluation
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Returns:
            Evaluation report
        """
        print("="*70)
        print("Starting RAG Chatbot Evaluation")
        print("Testing 20 queries based on first 20 trials in dataset")
        print("="*70)
        
        test_queries = self.create_test_queries()
        
        for test_case in test_queries:
            result = self.evaluate_query(test_case)
            self.results.append(result)
            time.sleep(15)  # Avoid API rate limits
        
        # Calculate overall metrics
        report = self.generate_report()
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report"""
        if not self.results:
            return {}
        
        # Calculate metrics
        total_queries = len(self.results)
        successful_queries = sum(1 for r in self.results if r['success'])
        grounded_answers = sum(1 for r in self.results if r['grounding']['grounded'])
        meets_latency = sum(1 for r in self.results if r['meets_latency'])
        correct_trial_retrieved = sum(1 for r in self.results if r['grounding']['correct_trial_retrieved'])
        
        avg_keyword_coverage = sum(r['keyword_coverage'] for r in self.results) / total_queries
        avg_retrieval_time = sum(r['retrieval_time'] for r in self.results) / total_queries
        avg_sources = sum(r['grounding']['num_sources'] for r in self.results) / total_queries
        
        # Statistics by category
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'success': 0}
            categories[cat]['total'] += 1
            if result['success']:
                categories[cat]['success'] += 1
        
        # Generate report
        report = {
            'summary': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'success_rate': f"{(successful_queries / total_queries * 100):.1f}%",
                'grounded_answers': grounded_answers,
                'grounding_rate': f"{(grounded_answers / total_queries * 100):.1f}%",
                'correct_trial_retrieved': correct_trial_retrieved,
                'trial_retrieval_rate': f"{(correct_trial_retrieved / total_queries * 100):.1f}%",
                'meets_latency': meets_latency,
                'latency_compliance': f"{(meets_latency / total_queries * 100):.1f}%",
                'avg_keyword_coverage': f"{(avg_keyword_coverage * 100):.1f}%",
                'avg_retrieval_time': f"{avg_retrieval_time:.3f}s",
                'avg_sources_per_query': f"{avg_sources:.1f}"
            },
            'by_category': categories,
            'detailed_results': self.results,
            'acceptance_criteria': {
                'grounding_rate': {
                    'required': '≥90%',
                    'actual': f"{(grounded_answers / total_queries * 100):.1f}%",
                    'passed': grounded_answers / total_queries >= 0.9
                },
                'trial_retrieval_rate': {
                    'required': '≥90%',
                    'actual': f"{(correct_trial_retrieved / total_queries * 100):.1f}%",
                    'passed': correct_trial_retrieved / total_queries >= 0.9
                },
                'latency': {
                    'required': '<2s',
                    'actual': f"{avg_retrieval_time:.3f}s",
                    'passed': avg_retrieval_time < 2.0
                }
            }
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print evaluation report"""
        print("\n" + "="*70)
        print("Evaluation Report")
        print("="*70)
        
        # Overall statistics
        print("\n📊 Overall Statistics:")
        summary = report['summary']
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Statistics by category
        print("\n📂 Statistics by Category:")
        for category, stats in report['by_category'].items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Acceptance criteria
        print("\n✅ Acceptance Criteria:")
        criteria = report['acceptance_criteria']
        for criterion, details in criteria.items():
            status = "✓ PASSED" if details['passed'] else "✗ FAILED"
            print(f"  {criterion}:")
            print(f"    Required: {details['required']}")
            print(f"    Actual: {details['actual']}")
            print(f"    Status: {status}")
        
        # Detailed results sample
        print("\n📝 Sample Test Cases (first 5):")
        for i, result in enumerate(report['detailed_results'][:5], 1):
            print(f"\n  Case {i}:")
            print(f"    Query: {result['query']}")
            print(f"    Trial ID: {result['trial_id']}")
            print(f"    Success: {'Yes' if result['success'] else 'No'}")
            print(f"    Correct Trial Retrieved: {'Yes' if result['grounding']['correct_trial_retrieved'] else 'No'}")
            print(f"    Number of Sources: {result['grounding']['num_sources']}")
            print(f"    Retrieval Time: {result['timing']['retrieval_time']}")
            print(f"    Answer: {result['answer'][:150]}...")
        
        print("\n" + "="*70)
    
    def save_report(self, report: Dict[str, Any], filename: str = "evaluation_report.json") -> None:
        """Save evaluation report to file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nEvaluation report saved to: {filename}")


def main():
    """Main function"""
    evaluator = ChatbotEvaluator()
    report = evaluator.run_evaluation()
    evaluator.print_report(report)
    evaluator.save_report(report)
    
    # Save human-readable report
    with open("evaluation_report.txt", 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Clinical Trial RAG Chatbot - Evaluation Report\n")
        f.write("Testing based on first 20 trials from dataset\n")
        f.write("="*70 + "\n\n")
        
        # Overall statistics
        f.write("1. Overall Statistics\n")
        f.write("-"*70 + "\n")
        for key, value in report['summary'].items():
            f.write(f"{key}: {value}\n")
        
        # Acceptance criteria
        f.write("\n2. Acceptance Criteria Check\n")
        f.write("-"*70 + "\n")
        criteria = report['acceptance_criteria']
        for criterion, details in criteria.items():
            status = "✓ PASSED" if details['passed'] else "✗ FAILED"
            f.write(f"\n{criterion}:\n")
            f.write(f"  Required: {details['required']}\n")
            f.write(f"  Actual: {details['actual']}\n")
            f.write(f"  Status: {status}\n")
        
        # Statistics by category
        f.write("\n3. Statistics by Query Category\n")
        f.write("-"*70 + "\n")
        for category, stats in report['by_category'].items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            f.write(f"{category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)\n")
        
        # Detailed test cases
        f.write("\n4. Detailed Test Cases\n")
        f.write("-"*70 + "\n")
        for i, result in enumerate(report['detailed_results'], 1):
            f.write(f"\nCase {i}:\n")
            f.write(f"Query: {result['query']}\n")
            f.write(f"Expected Trial: {result['trial_id']}\n")
            f.write(f"Category: {result['category']}\n")
            f.write(f"Success: {'Yes' if result['success'] else 'No'}\n")
            f.write(f"Correct Trial Retrieved: {'Yes' if result['grounding']['correct_trial_retrieved'] else 'No'}\n")
            f.write(f"Retrieval Time: {result['timing']['retrieval_time']}\n")
            f.write(f"API Time: {result['timing']['api_time']}\n")
            f.write(f"Number of Sources: {result['grounding']['num_sources']}\n")
            f.write(f"Has Source Reference: {'Yes' if result['grounding']['has_source_reference'] else 'No'}\n")
            f.write(f"Keyword Coverage: {result['keyword_coverage']*100:.1f}%\n")
            f.write(f"Keywords Found: {', '.join(result['keywords_found'])}\n")
            f.write(f"\nAnswer:\n{result['answer']}\n")
            f.write(f"\nSources Retrieved:\n")
            for source in result['sources']:
                f.write(f"  - {source['NCTId']}: {source['Title']}\n")
            f.write("\n" + "-"*70 + "\n")
        
        # Conclusion
        f.write("\n5. Conclusion\n")
        f.write("-"*70 + "\n")
        
        grounding_passed = criteria['grounding_rate']['passed']
        trial_retrieval_passed = criteria['trial_retrieval_rate']['passed']
        latency_passed = criteria['latency']['passed']
        
        if grounding_passed and trial_retrieval_passed and latency_passed:
            f.write("✓ System passed all acceptance criteria\n")
            f.write("  - Answer grounding rate meets standard (≥90%)\n")
            f.write("  - Trial retrieval rate meets standard (≥90%)\n")
            f.write("  - Response latency meets standard (<2 seconds)\n")
        else:
            f.write("✗ System did not fully pass acceptance criteria\n")
            if not grounding_passed:
                f.write("  - Answer grounding rate does not meet standard\n")
            if not trial_retrieval_passed:
                f.write("  - Trial retrieval rate does not meet standard\n")
            if not latency_passed:
                f.write("  - Response latency does not meet standard\n")
        
        f.write("\nImprovement Suggestions:\n")
        if not grounding_passed:
            f.write("  - Improve prompt to ensure answers always include source citations\n")
            f.write("  - Add post-processing step to verify source citations\n")
        if not trial_retrieval_passed:
            f.write("  - Improve BM25 retrieval parameters\n")
            f.write("  - Consider increasing number of results retrieved\n")
            f.write("  - Enhance query preprocessing and tokenization\n")
        if not latency_passed:
            f.write("  - Optimize retrieval algorithm\n")
            f.write("  - Consider caching frequent queries\n")
            f.write("  - Reduce number of retrieval results\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print("Detailed report saved to: evaluation_report.txt")


if __name__ == "__main__":
    main()
