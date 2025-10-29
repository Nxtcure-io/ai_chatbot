"""
Data Indexer - Create BM25 index for clinical trials data
"""
import json
import time
import pickle
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class TrialIndexer:
    """Clinical Trial Data Indexer - Using BM25"""
    
    def __init__(self, json_file: str):
        """
        Initialize indexer
        
        Args:
            json_file: Path to clinical trials JSON file
        """
        self.json_file = json_file
        self.trials = []
        self.bm25 = None
        self.tokenized_corpus = []
        
    def load_trials(self) -> List[Dict[str, Any]]:
        """Load clinical trials data"""
        print(f"Loading clinical trials data from {self.json_file}...")
        start_time = time.time()
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.trials = json.load(f)
        
        elapsed_time = time.time() - start_time
        print(f"Loaded {len(self.trials)} clinical trials in {elapsed_time:.2f} seconds")
        return self.trials
    
    def create_trial_text(self, trial: Dict[str, Any]) -> str:
        """
        Create searchable text for each clinical trial
        
        Args:
            trial: Clinical trial data dictionary
            
        Returns:
            Formatted text string
        """
        parts = []
        
        # Basic information
        if trial.get('NCTId'):
            parts.append(f"Trial ID: {trial['NCTId']}")
        if trial.get('BriefTitle'):
            parts.append(f"Title: {trial['BriefTitle']}")
        if trial.get('OfficialTitle'):
            parts.append(f"Official Title: {trial['OfficialTitle']}")
        
        # Study status and type
        if trial.get('OverallStatus'):
            parts.append(f"Status: {trial['OverallStatus']}")
        if trial.get('Phase'):
            parts.append(f"Phase: {trial['Phase']}")
        if trial.get('StudyType'):
            parts.append(f"Study Type: {trial['StudyType']}")
        
        # Conditions and interventions
        if trial.get('Conditions'):
            parts.append(f"Conditions: {trial['Conditions']}")
        if trial.get('Interventions'):
            parts.append(f"Interventions: {trial['Interventions']}")
        
        # Eligibility criteria (important!)
        if trial.get('EligibilityCriteria'):
            criteria = trial['EligibilityCriteria'].replace('\n', ' ')
            parts.append(f"Eligibility: {criteria}")
        
        # Participant information
        if trial.get('HealthyVolunteers'):
            parts.append(f"Healthy Volunteers: {trial['HealthyVolunteers']}")
        if trial.get('Sex'):
            parts.append(f"Sex: {trial['Sex']}")
        if trial.get('MinimumAge'):
            parts.append(f"Minimum Age: {trial['MinimumAge']}")
        if trial.get('MaximumAge'):
            parts.append(f"Maximum Age: {trial['MaximumAge']}")
        if trial.get('StandardAges'):
            parts.append(f"Ages: {trial['StandardAges']}")
        
        # Contact information
        if trial.get('PrimaryContactName'):
            parts.append(f"Contact: {trial['PrimaryContactName']}")
        if trial.get('PrimaryContactEmail'):
            parts.append(f"Email: {trial['PrimaryContactEmail']}")
        
        # Investigator information
        if trial.get('PrincipalInvestigatorName'):
            parts.append(f"PI: {trial['PrincipalInvestigatorName']}")
        if trial.get('PrincipalInvestigatorAffiliation'):
            parts.append(f"Affiliation: {trial['PrincipalInvestigatorAffiliation']}")
        
        # Location
        if trial.get('USLocations'):
            parts.append(f"Locations: {trial['USLocations']}")
        if trial.get('Country'):
            parts.append(f"Country: {trial['Country']}")
        
        # Dates
        if trial.get('StartDate'):
            parts.append(f"Start Date: {trial['StartDate']}")
        if trial.get('CompletionDate'):
            parts.append(f"Completion Date: {trial['CompletionDate']}")
        
        # Outcomes
        if trial.get('PrimaryOutcomes'):
            parts.append(f"Primary Outcomes: {trial['PrimaryOutcomes']}")
        if trial.get('SecondaryOutcomes'):
            parts.append(f"Secondary Outcomes: {trial['SecondaryOutcomes']}")
        
        return " ".join(parts)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple text tokenization
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split by whitespace
        tokens = text.lower().split()
        # Remove punctuation
        tokens = [''.join(c for c in token if c.isalnum()) for token in tokens]
        # Remove empty strings
        tokens = [t for t in tokens if t]
        return tokens
    
    def build_index(self) -> None:
        """Build BM25 index"""
        print("="*70)
        print("Building BM25 Index...")
        print("="*70)
        
        # 1. Load data
        self.load_trials()
        
        # 2. Create document corpus
        print("\nCreating document corpus...")
        start_time = time.time()
        corpus = []
        for i, trial in enumerate(self.trials):
            text = self.create_trial_text(trial)
            corpus.append(text)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(self.trials)} trials...")
        
        elapsed_time = time.time() - start_time
        print(f"Corpus creation completed in {elapsed_time:.2f} seconds")
        
        # 3. Tokenization
        print("\nTokenizing documents...")
        start_time = time.time()
        self.tokenized_corpus = [self.tokenize(doc) for doc in corpus]
        elapsed_time = time.time() - start_time
        print(f"Tokenization completed in {elapsed_time:.2f} seconds")
        
        # 4. Build BM25 index
        print("\nBuilding BM25 index...")
        start_time = time.time()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        elapsed_time = time.time() - start_time
        print(f"BM25 index built in {elapsed_time:.2f} seconds")
        
        # 5. Save index
        print("\nSaving index...")
        start_time = time.time()
        self.save_index()
        elapsed_time = time.time() - start_time
        print(f"Index saved in {elapsed_time:.2f} seconds")
        
        print("\n" + "="*70)
        print("Index building completed!")
        print("="*70)
    
    def save_index(self) -> None:
        """Save BM25 index and trials data"""
        # Save BM25 index
        with open('bm25_index.pkl', 'wb') as f:
            pickle.dump(self.bm25, f)
        print("BM25 index saved to: bm25_index.pkl")
        
        # Save tokenized corpus
        with open('tokenized_corpus.pkl', 'wb') as f:
            pickle.dump(self.tokenized_corpus, f)
        print("Tokenized corpus saved to: tokenized_corpus.pkl")
        
        # Save trials data (for quick loading)
        with open('trials_data.pkl', 'wb') as f:
            pickle.dump(self.trials, f)
        print("Trials data saved to: trials_data.pkl")


def main():
    """Main function"""
    print("="*70)
    print("Clinical Trials Data Indexer (BM25)")
    print("="*70)
    print("\nAdvantages of using BM25:")
    print("  ✓ No need to download models")
    print("  ✓ Fully offline operation")
    print("  ✓ Fast performance")
    print("  ✓ Low memory usage")
    print("  ✓ Good for keyword matching")
    print("\n" + "="*70 + "\n")
    
    indexer = TrialIndexer("us_recruiting_trials_detailed_20251026_114355.json")
    indexer.build_index()
    
    print("\n" + "="*70)
    print("You can now run the chatbot:")
    print("  streamlit run app.py")
    print("="*70)


if __name__ == "__main__":
    main()
