import glob
import sys

files = glob.glob('documents/*.txt')

for fpath in files:
    print(f"Checking {fpath}...", end=' ')
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            f.read()
        print("Valid UTF-8")
    except UnicodeDecodeError:
        print("NOT UTF-8")
        # Try cp1252
        try:
            with open(fpath, 'r', encoding='cp1252') as f:
                f.read()
            print(f"  -> Valid CP1252")
        except:
            print(f"  -> Unknown encoding")
    sys.stdout.flush()
