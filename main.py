import argparse
from semantic_diff import SemanticDiff
from pipeline import run_pipeline
import sys
import logging

# Suppress logs
logging.basicConfig(level=logging.ERROR)

def main():
    parser = argparse.ArgumentParser(description='Semantic Diff Tool')
    parser.add_argument('file1', help='Path to the first file')
    parser.add_argument('file2', help='Path to the second file')
    parser.add_argument('--threshold', type=float, default=None, help='Similarity threshold (0.0 to 1.0). If not provided, it will be computed automatically.')
    parser.add_argument('--split-markdown', action='store_true', help='Split markdown files by headers and diff sections individually.')
    parser.add_argument('--output', help='Path to save the output.')
    
    args = parser.parse_args()
    
    if args.split_markdown:
        run_pipeline(args.file1, args.file2, threshold=args.threshold, output_file=args.output)
        return

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

    output_lines = []
    for action, s1, s2, score in alignment:
        s1 = clean_text(s1)
        s2 = clean_text(s2)
        if action == 'MATCH':
            line = f"  [MATCH] ({score:.2f}): {s1}"
            print(line)
            output_lines.append(line)
        elif action == 'MODIFIED':
            line1 = f"~ [MODIFIED] ({score:.2f}):"
            line2 = f"    < {s1}"
            line3 = f"    > {s2}"
            print(line1)
            print(line2)
            print(line3)
            output_lines.append(line1)
            output_lines.append(line2)
            output_lines.append(line3)
        elif action == 'DELETE':
            line = f"- [DELETE]: {s1}"
            print(line)
            output_lines.append(line)
        elif action == 'INSERT':
            line = f"+ [INSERT]: {s2}"
            print(line)
            output_lines.append(line)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"\nOutput saved to {args.output}")

if __name__ == '__main__':
    main()
