import argparse
from semantic_diff import SemanticDiff
import sys
import logging

# Suppress logs
logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description='Semantic Diff Tool')
    parser.add_argument('file1', help='Path to the first file')
    parser.add_argument('file2', help='Path to the second file')
    parser.add_argument('--threshold', type=float, default=None, help='Similarity threshold (0.0 to 1.0). If not provided, it will be computed automatically.')
    
    args = parser.parse_args()
    
    if args.threshold is not None:
        print(f"Comparing {args.file1} and {args.file2} with threshold {args.threshold}...")
    else:
        print(f"Comparing {args.file1} and {args.file2} with auto-computed threshold...")
    
    diff_tool = SemanticDiff(similarity_threshold=args.threshold)
    alignment = diff_tool.diff_files(args.file1, args.file2)
    
    if diff_tool.last_threshold_stats:
        print("\n--- Threshold Statistics ---")
        stats = diff_tool.last_threshold_stats
        print(f"Computed Threshold: {stats['computed_threshold']:.4f}")
        if 'mean' in stats:
            print(f"Mean Max Sim: {stats['mean']:.4f}")
            print(f"Std Dev: {stats['std']:.4f}")
            print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print("\n--- Semantic Diff Results ---\n")
    
    def clean_text(text):
        if text is None: return ""
        # Replace common non-ascii chars with ascii equivalents
        text = text.replace('—', '--').replace('’', "'").replace('“', '"').replace('”', '"')
        # Replace newlines and carriage returns to prevent terminal jumbling
        return text.replace('\n', ' ').replace('\r', ' ')

    for action, s1, s2, score in alignment:
        s1 = clean_text(s1)
        s2 = clean_text(s2)
        if action == 'MATCH':
            if s1 == s2:
                print(f"  [MATCH=] ({score:.2f}): {s1}")
            else:
                print(f"  [MATCH] ({score:.2f}): {s1}")
                print(f"    < {s1}")
                print(f"    > {s2}")
        elif action == 'MODIFIED':
            print(f"~ [MODIFIED] ({score:.2f}):")
            print(f"    < {s1}")
            print(f"    > {s2}")
        elif action == 'DELETE':
            print(f"- [DELETE]: {s1}")
        elif action == 'INSERT':
            print(f"+ [INSERT]: {s2}")

if __name__ == '__main__':
    main()
