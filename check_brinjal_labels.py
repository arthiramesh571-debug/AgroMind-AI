import pyarrow.parquet as pq
from collections import Counter

parquet_path = r"C:\Users\Arthi.R\Downloads\train-00000-of-00001.parquet"

parquet_file = pq.ParquetFile(parquet_path)
counts = Counter()
examples = {}

for batch in parquet_file.iter_batches(batch_size=2000):
    for row in batch.to_pylist():
        label = row["label"]
        counts[label] += 1
        examples.setdefault(label, row["image"]["path"])

print("LABEL COUNTS:")
for label, count in sorted(counts.items()):
    print(f"Label {label}: {count} images")

print("\nSAMPLE FILENAMES:")
for label, filename in sorted(examples.items()):
    print(f"Label {label}: {filename}")
