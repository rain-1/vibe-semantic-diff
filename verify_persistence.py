import os
import sqlite3
from semantic_diff import SemanticDiff

def create_dummy_files():
    with open('dummy1.txt', 'w', encoding='utf-8') as f:
        f.write("The quick brown fox jumps over the lazy dog.\nThis is a unique sentence for file 1.")
    with open('dummy2.txt', 'w', encoding='utf-8') as f:
        f.write("The quick brown fox jumps over the lazy dog.\nThis is a unique sentence for file 2.")
    with open('dummy3.txt', 'w', encoding='utf-8') as f:
        f.write("This is a unique sentence for file 1.\nSome other content.")

def verify_db_content():
    db_path = 'embeddings.db'
    if not os.path.exists(db_path):
        print("FAIL: Database file not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM embeddings")
    count = cursor.fetchone()[0]
    print(f"Database contains {count} embeddings.")
    
    cursor.execute("SELECT text, doc_name FROM embeddings")
    rows = cursor.fetchall()
    print("Sample entries:")
    for row in rows[:5]:
        print(f"  - {row}")
    
    conn.close()
    
    if count > 0:
        print("PASS: Embeddings stored.")
    else:
        print("FAIL: No embeddings stored.")

def run_verification():
    create_dummy_files()
    
    # Use absolute paths
    f1 = os.path.abspath('dummy1.txt')
    f2 = os.path.abspath('dummy2.txt')
    f3 = os.path.abspath('dummy3.txt')
    
    diff = SemanticDiff()
    
    print("\n--- Running Diff 1 vs 2 ---")
    diff.diff_files(f1, f2)
    
    verify_db_content()
    
    print("\n--- Running Diff 1 vs 3 (Should trigger match alert) ---")
    # This should trigger an alert because "This is a unique sentence for file 1." was stored from dummy1.txt
    # and now we see it in dummy3.txt (or similar).
    # Wait, dummy3 has "This is a unique sentence for file 1." which matches dummy1.
    diff.diff_files(f1, f3)

    # Clean up
    # os.remove('dummy1.txt')
    # os.remove('dummy2.txt')
    # os.remove('dummy3.txt')
    # os.remove('embeddings.db') # Keep for inspection

if __name__ == '__main__':
    run_verification()
