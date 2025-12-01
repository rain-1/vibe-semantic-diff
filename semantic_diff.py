import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticDiff:
    def __init__(self, model_name='all-MiniLM-L6-v2', similarity_threshold=None):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.last_threshold_stats = {}

    def split_sentences(self, text):
        # Simple regex-based sentence splitting for now
        # This can be improved with NLTK or Spacy
        # Combined regex to split by sentence-ending punctuation followed by space, or by newline
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s+|\n', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_embeddings(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True)

    def compute_similarity_matrix(self, emb1, emb2):
        # Compute cosine similarity
        return util.cos_sim(emb1, emb2).cpu().numpy()

    def align_sentences(self, sentences1, sentences2):
        if not sentences1 or not sentences2:
            return []

        emb1 = self.get_embeddings(sentences1)
        emb2 = self.get_embeddings(sentences2)
        sim_matrix = self.compute_similarity_matrix(emb1, emb2)
        
        n = len(sentences1)
        m = len(sentences2)
        
        # Determine threshold
        threshold = self.similarity_threshold
        if threshold is None:
            # Compute dynamic threshold
            # We look at the distribution of the best match for each sentence
            max_sim_1 = np.max(sim_matrix, axis=1)
            max_sim_2 = np.max(sim_matrix, axis=0)
            all_max = np.concatenate((max_sim_1, max_sim_2))
            
            mean_max = np.mean(all_max)
            std_max = np.std(all_max)
            
            # Heuristic: Threshold is mean - 0.5 * std
            # This assumes that "matches" are the dominant mode, or at least significant
            threshold = float(mean_max) - 0.5 * float(std_max)
            
            # Ensure threshold isn't too high (e.g. if documents are identical)
            # Cap at 0.95 unless mean is very low? 
            # Actually, if documents are identical, mean=1.0, std=0.0 -> threshold=1.0.
            # We should subtract a small epsilon to handle float noise.
            threshold = min(threshold, 0.95)
            
            self.last_threshold_stats = {
                'mean': float(mean_max),
                'std': float(std_max),
                'min': float(np.min(all_max)),
                'max': float(np.max(all_max)),
                'computed_threshold': threshold
            }
        else:
            self.last_threshold_stats = {'computed_threshold': threshold}
        
        # DP Initialization
        # dp[i][j] stores the max score
        dp = np.zeros((n + 1, m + 1))
        
        # Pointers to reconstruct the path: 0=diag, 1=up, 2=left
        pointers = np.zeros((n + 1, m + 1), dtype=int)
        
        gap_penalty = 0.0 # Maybe small positive to encourage skipping bad matches? Or 0.
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Optimization: Exact string match
                if sentences1[i-1] == sentences2[j-1]:
                    sim_matrix[i-1][j-1] = 1.0
                
                score_match = dp[i-1][j-1] + sim_matrix[i-1][j-1]
                score_skip_1 = dp[i-1][j] - gap_penalty
                score_skip_2 = dp[i][j-1] - gap_penalty
                
                # We prefer matching if it's good enough
                # But if similarity is low, maybe skipping is better?
                # For now, standard LCS-like logic but with weights
                
                best_score = max(score_match, score_skip_1, score_skip_2)
                dp[i][j] = best_score
                
                if best_score == score_match:
                    pointers[i][j] = 0
                elif best_score == score_skip_1:
                    pointers[i][j] = 1
                else:
                    pointers[i][j] = 2
                    
        # Backtrack
        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pointers[i][j] == 0:
                # Match
                sim = sim_matrix[i-1][j-1]
                # Force match if strings are identical or sim is high enough
                if sentences1[i-1] == sentences2[j-1] or sim >= threshold:
                    alignment.append(('MATCH', sentences1[i-1], sentences2[j-1], sim))
                else:
                    # Even if it was the "best" path, if it's below threshold, treat as diff
                    # This is a bit tricky. The DP found the best path assuming we take it.
                    # Maybe we should treat it as a modification?
                    alignment.append(('MODIFIED', sentences1[i-1], sentences2[j-1], sim))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or pointers[i][j] == 1):
                # Deletion from doc1
                alignment.append(('DELETE', sentences1[i-1], None, 0))
                i -= 1
            else:
                # Insertion into doc2
                alignment.append(('INSERT', None, sentences2[j-1], 0))
                j -= 1
                
        return list(reversed(alignment))

    def diff_files(self, file1_path, file2_path):
        with open(file1_path, 'r', encoding='utf-8') as f:
            text1 = f.read()
        with open(file2_path, 'r', encoding='utf-8') as f:
            text2 = f.read()
            
        return self.diff_files_from_text(text1, text2)

    def diff_files_from_text(self, text1, text2):
        sent1 = self.split_sentences(text1)
        sent2 = self.split_sentences(text2)
        return self.align_sentences(sent1, sent2)
