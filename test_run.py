from semantic_diff import SemanticDiff
import sys

def run_test():
    diff = SemanticDiff()
    # Test 1 vs 3
    print("Running diff 1 vs 3...")
    res1 = diff.diff_files('documents/letter-1.txt', 'documents/letter-3.txt')
    
    with open('result_1_3.txt', 'w', encoding='utf-8') as f:
        for action, s1, s2, score in res1:
            if action == 'MATCH':
                f.write(f"  [MATCH] ({score:.2f}): {s1}\n")
            elif action == 'MODIFIED':
                f.write(f"~ [MODIFIED] ({score:.2f}):\n")
                f.write(f"    < {s1}\n")
                f.write(f"    > {s2}\n")
            elif action == 'DELETE':
                f.write(f"- [DELETE]: {s1}\n")
            elif action == 'INSERT':
                f.write(f"+ [INSERT]: {s2}\n")

    # Test 1 vs 2
    print("Running diff 1 vs 2...")
    res2 = diff.diff_files('documents/letter-1.txt', 'documents/letter-2.txt')
    with open('result_1_2.txt', 'w', encoding='utf-8') as f:
        for action, s1, s2, score in res2:
            if action == 'MATCH':
                f.write(f"  [MATCH] ({score:.2f}): {s1}\n")
            elif action == 'MODIFIED':
                f.write(f"~ [MODIFIED] ({score:.2f}):\n")
                f.write(f"    < {s1}\n")
                f.write(f"    > {s2}\n")
            elif action == 'DELETE':
                f.write(f"- [DELETE]: {s1}\n")
            elif action == 'INSERT':
                f.write(f"+ [INSERT]: {s2}\n")

if __name__ == '__main__':
    run_test()
