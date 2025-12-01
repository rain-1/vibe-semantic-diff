with open('documents/letter-3.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if 'situation' in line:
            print(f"Line: {line}")
            print(f"Repr: {ascii(line)}")
