
import csv, os
from collections import Counter
CSV_FILE = "punch_data.csv"
c = Counter()
with open(CSV_FILE, newline='') as f:
    reader = csv.reader(f)
    headers = next(reader)
    for r in reader:
        if r:
            c[r[-1]] += 1
print("Counts:", dict(c))
