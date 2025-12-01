import re

def split_markdown(filepath):
    """
    Splits a markdown file into sections based on headers.
    Returns a list of dictionaries: {'title': str, 'content': str, 'level': int}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sections = []
    current_title = "Preamble"
    current_level = 0
    current_content = []

    header_pattern = re.compile(r'^(#+)\s+(.*)')

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Save previous section
            if current_content or current_title != "Preamble":
                sections.append({
                    'title': current_title,
                    'content': "".join(current_content).strip(),
                    'level': current_level
                })
            
            # Start new section
            current_level = len(match.group(1))
            current_title = match.group(2).strip()
            current_content = [line] # Include header in content? Maybe yes for context.
        else:
            current_content.append(line)

    # Append last section
    if current_content:
        sections.append({
            'title': current_title,
            'content': "".join(current_content).strip(),
            'level': current_level
        })

    return sections
