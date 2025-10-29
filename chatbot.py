"""
RAG Chatbot Core Logic - Using BM25 Retrieval
"""
import os
import json
import time
import pickle
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from openai import OpenAI


class RAGChatbot:
    """Clinical Trial RAG Chatbot based on BM25 Retrieval-Augmented Generation"""
    
    def __init__(self):
        """Initialize chatbot"""
        # Set HuggingFace Token
        os.environ["HF_TOKEN"] = ""
        
        # Initialize OpenAI client (for HuggingFace)
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
        )
        
        # Load BM25 index and data
        print("Loading BM25 index...")
        start_time = time.time()
        
        with open('bm25_index.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)
        
        with open('tokenized_corpus.pkl', 'rb') as f:
            self.tokenized_corpus = pickle.load(f)
        
        with open('trials_data.pkl', 'rb') as f:
            self.trials = pickle.load(f)
        
        elapsed = time.time() - start_time
        print(f"Index loaded in {elapsed:.2f} seconds")
        print(f"Loaded {len(self.trials)} clinical trials")
        
        # Time statistics
        self.stats = {
            'total_queries': 0,
            'total_retrieval_time': 0,
            'total_api_time': 0,
            'total_time': 0
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple text tokenization (consistent with indexer)
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        tokens = text.lower().split()
        tokens = [''.join(c for c in token if c.isalnum()) for token in tokens]
        tokens = [t for t in tokens if t]
        return tokens
    
    def retrieve_relevant_trials(self, query: str, n_results: int = 10) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant clinical trials using BM25
        
        Args:
            query: User query
            n_results: Number of results to return
            
        Returns:
            (List of relevant trials, retrieval time)
        """
        start_time = time.time()
        
        # Tokenize query
        tokenized_query = self.tokenize(query)
        
        # BM25 retrieval
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k results
        top_n_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_results]
        
        retrieval_time = time.time() - start_time
        
        # Build results
        relevant_trials = []
        for idx in top_n_indices:
            trial = self.trials[idx]
            trial_info = {
                'trial': trial,
                'score': float(scores[idx]),
                'index': idx
            }
            relevant_trials.append(trial_info)
        
        return relevant_trials, retrieval_time
    
    def format_context(self, relevant_trials: List[Dict]) -> str:
        """
        Format retrieved trials as context
        
        Args:
            relevant_trials: List of relevant trials
            
        Returns:
            Formatted context string
        """
        if not relevant_trials:
            return "No relevant clinical trials found."
        
        context_parts = []
        for i, item in enumerate(relevant_trials, 1):
            trial = item['trial']
            
            context = f"\n[Trial {i} - {trial.get('NCTId', 'Unknown')}]\n"
            context += f"Title: {trial.get('BriefTitle', 'N/A')}\n"
            
            if trial.get('OfficialTitle'):
                context += f"Official Title: {trial.get('OfficialTitle')}\n"
            
            context += f"Status: {trial.get('OverallStatus', 'N/A')}\n"
            context += f"Phase: {trial.get('Phase', 'N/A')}\n"
            context += f"Study Type: {trial.get('StudyType', 'N/A')}\n"
            context += f"Conditions: {trial.get('Conditions', 'N/A')}\n"
            
            if trial.get('Interventions'):
                context += f"Interventions: {trial.get('Interventions')}\n"
            
            if trial.get('EligibilityCriteria'):
                # Limit length
                criteria = trial.get('EligibilityCriteria')[:500]
                context += f"Eligibility (excerpt): {criteria}...\n"
            
            if trial.get('HealthyVolunteers'):
                context += f"Healthy Volunteers: {trial.get('HealthyVolunteers')}\n"
            
            if trial.get('Sex'):
                context += f"Sex: {trial.get('Sex')}\n"
            
            if trial.get('MinimumAge'):
                context += f"Age Range: {trial.get('MinimumAge', 'N/A')} - {trial.get('MaximumAge', 'N/A')}\n"
            
            if trial.get('USLocations'):
                context += f"Locations: {trial.get('USLocations')}\n"
            
            if trial.get('PrimaryContactEmail'):
                context += f"Contact Email: {trial.get('PrimaryContactEmail')}\n"
            
            if trial.get('PrimaryContactPhone'):
                context += f"Contact Phone: {trial.get('PrimaryContactPhone')}\n"
            
            if trial.get('StartDate'):
                context += f"Start Date: {trial.get('StartDate')}\n"
            
            context_parts.append(context)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> Tuple[str, float]:
        """
        Generate answer using LLM
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            (Answer, API call time)
        """
        start_time = time.time()
        
        # Build prompt
        system_prompt = """You are a professional clinical trial information assistant. Your tasks:
1. Answer questions based ONLY on the provided clinical trial data
2. Always cite specific Trial IDs (NCTId) as sources
3. If the provided data doesn't contain relevant information, clearly state "Based on the provided data, I cannot find relevant information"
4. Do not fabricate or speculate any information
5. Use concise, professional language to answer
6. If the question involves multiple trials, list all relevant trials with their IDs"""

        user_prompt = f"""Answer the question based on the following clinical trial data.

Clinical Trial Data:
{context}

User Question: {query}

Please provide an answer based on the above data and cite specific Trial IDs. If the data doesn't contain relevant information, please state it clearly."""

        try:
            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-3.2-1B-Instruct:novita",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            answer = completion.choices[0].message.content
            api_time = time.time() - start_time
            
            return answer, api_time
            
        except Exception as e:
            api_time = time.time() - start_time
            return f"Sorry, error generating answer: {str(e)}", api_time
    
    def chat(self, query: str) -> Dict[str, Any]:
        """
        Process user query
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing answer, sources, and timing statistics
        """
        total_start_time = time.time()
        
        # 1. Retrieve relevant trials
        relevant_trials, retrieval_time = self.retrieve_relevant_trials(query)
        
        # 2. Format context
        context = self.format_context(relevant_trials)
        
        # 3. Generate answer
        answer, api_time = self.generate_answer(query, context)
        
        # 4. Calculate total time
        total_time = time.time() - total_start_time
        
        # 5. Update statistics
        self.stats['total_queries'] += 1
        self.stats['total_retrieval_time'] += retrieval_time
        self.stats['total_api_time'] += api_time
        self.stats['total_time'] += total_time
        
        # 6. Prepare source citations
        sources = []
        for item in relevant_trials:
            trial = item['trial']
            sources.append({
                'NCTId': trial.get('NCTId', 'Unknown'),
                'Title': trial.get('BriefTitle', 'N/A'),
                'Score': f"{item['score']:.2f}",
                'Relevance': f"{min(item['score'] * 10, 100):.1f}%"  # Normalize for display
            })
        
        return {
            'answer': answer,
            'sources': sources,
            'timing': {
                'retrieval_time': f"{retrieval_time:.3f}s",
                'api_time': f"{api_time:.3f}s",
                'total_time': f"{total_time:.3f}s"
            },
            'context': context  # For debugging
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if self.stats['total_queries'] == 0:
            return self.stats
        
        return {
            'total_queries': self.stats['total_queries'],
            'avg_retrieval_time': f"{self.stats['total_retrieval_time'] / self.stats['total_queries']:.3f}s",
            'avg_api_time': f"{self.stats['total_api_time'] / self.stats['total_queries']:.3f}s",
            'avg_total_time': f"{self.stats['total_time'] / self.stats['total_queries']:.3f}s",
            'total_retrieval_time': f"{self.stats['total_retrieval_time']:.3f}s",
            'total_api_time': f"{self.stats['total_api_time']:.3f}s",
            'total_time': f"{self.stats['total_time']:.3f}s"
        }


def main():
    """Test chatbot"""
    print("Initializing chatbot...")
    chatbot = RAGChatbot()
    
    # Test queries
    test_queries = [
        "Are there any clinical trials for PTSD?",
        "Which trials accept healthy volunteers?",
        "Are there any studies for adolescent obesity?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print(f"{'='*60}")
        
        result = chatbot.chat(query)
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source['NCTId']}: {source['Title']} (BM25 Score: {source['Score']})")
        
        print(f"\nTiming Statistics:")
        print(f"  Retrieval Time: {result['timing']['retrieval_time']}")
        print(f"  API Call Time: {result['timing']['api_time']}")
        print(f"  Total Time: {result['timing']['total_time']}")
    
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    stats = chatbot.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
