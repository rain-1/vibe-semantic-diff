import sqlite3
import numpy as np
import io
import datetime

class DatabaseManager:
    def __init__(self, db_path='embeddings.db'):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector BLOB,
                    text TEXT,
                    doc_name TEXT,
                    doc_path TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.commit()

    def store_embedding(self, vector, text, doc_name, doc_path):
        """
        Store an embedding in the database.
        vector: numpy array
        text: str
        doc_name: str
        doc_path: str
        """
        # Convert numpy array to bytes
        vector_bytes = vector.tobytes()
        created_at = datetime.datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO embeddings (vector, text, doc_name, doc_path, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (vector_bytes, text, doc_name, doc_path, created_at))
            conn.commit()

    def find_matches(self, vector, threshold=0.99):
        """
        Find matches for a given vector in the database.
        Returns a list of (text, doc_name, doc_path, score) tuples.
        """
        matches = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT vector, text, doc_name, doc_path FROM embeddings')
            rows = cursor.fetchall()
            
            for row_vector_bytes, row_text, row_doc_name, row_doc_path in rows:
                # Convert bytes back to numpy array
                row_vector = np.frombuffer(row_vector_bytes, dtype=vector.dtype)
                
                # Compute cosine similarity
                # Ensure vectors are normalized if not already? 
                # SentenceTransformer embeddings are usually normalized.
                # Cosine similarity = (A . B) / (||A|| * ||B||)
                # If normalized, just dot product.
                
                dot_product = np.dot(vector, row_vector)
                norm_a = np.linalg.norm(vector)
                norm_b = np.linalg.norm(row_vector)
                
                if norm_a == 0 or norm_b == 0:
                    score = 0
                else:
                    score = dot_product / (norm_a * norm_b)
                
                if score >= threshold:
                    matches.append((row_text, row_doc_name, row_doc_path, score))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[3], reverse=True)
        return matches
