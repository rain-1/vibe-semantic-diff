import os
from markdown_splitter import split_markdown
from semantic_diff import SemanticDiff
import sys

def clean_text(text):
    if text is None: return ""
    text = text.replace('—', '--').replace('’', "'").replace('“', '"').replace('”', '"')
    return text.replace('\n', ' ').replace('\r', ' ')

def run_pipeline(file1, file2, threshold=None, output_file=None):
    print(f"Splitting {file1}...")
    sections_1 = split_markdown(file1)
    print(f"Splitting {file2}...")
    sections_2 = split_markdown(file2)

    print(f"Found {len(sections_1)} sections in file 1 and {len(sections_2)} sections in file 2.")

    # Initialize Diff Tool
    diff_tool = SemanticDiff(similarity_threshold=threshold)

    # Output buffer
    output = []

    # Simple matching strategy: Match by exact title
    # We'll iterate through sections_1 and look for counterparts in sections_2
    
    # Map file2 sections by title for quick lookup
    # Handling duplicate titles? For now, assume unique or take first.
    map_2 = {s['title']: s for s in sections_2}
    used_titles_2 = set()

    for sec1 in sections_1:
        title = sec1['title']
        content1 = sec1['content']
        
        output.append(f"\n{'='*40}")
        output.append(f"SECTION: {title}")
        output.append(f"{'='*40}\n")

        if title in map_2:
            sec2 = map_2[title]
            used_titles_2.add(title)
            content2 = sec2['content']
            
            # Run Semantic Diff
            print(f"Diffing section: {title}...")
            # Pass file paths so embeddings are stored with correct metadata
            alignment = diff_tool.diff_files_from_text(content1, content2, file1_path=file1, file2_path=file2)
            
            for action, s1, s2, score in alignment:
                s1 = clean_text(s1)
                s2 = clean_text(s2)
                if action == 'MATCH':
                    output.append(f"  [MATCH] ({score:.2f}): {s1}")
                elif action == 'MODIFIED':
                    output.append(f"~ [MODIFIED] ({score:.2f}):")
                    output.append(f"    < {s1}")
                    output.append(f"    > {s2}")
                elif action == 'DELETE':
                    output.append(f"- [DELETE]: {s1}")
                elif action == 'INSERT':
                    output.append(f"+ [INSERT]: {s2}")
        else:
            # Section deleted or renamed
            print(f"Section '{title}' not found in file 2.")
            output.append(f"!!! SECTION '{title}' REMOVED OR RENAMED !!!")
            # Dump content as deleted?
            # For brevity, maybe just say it's gone. Or dump it.
            # Let's dump lines as deleted
            for line in content1.split('\n'):
                if line.strip():
                    output.append(f"- [DELETE_SECTION]: {clean_text(line)}")

    # Check for new sections in file 2
    for sec2 in sections_2:
        if sec2['title'] not in used_titles_2:
            title = sec2['title']
            print(f"New section '{title}' found in file 2.")
            output.append(f"\n{'='*40}")
            output.append(f"NEW SECTION: {title}")
            output.append(f"{'='*40}\n")
            for line in sec2['content'].split('\n'):
                if line.strip():
                    output.append(f"+ [INSERT_SECTION]: {clean_text(line)}")

    # Write to file or stdout
    final_output = "\n".join(output)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"\nPipeline finished. Output written to {output_file}")
    else:
        print(final_output)
