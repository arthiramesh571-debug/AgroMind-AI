import json
from pathlib import Path
import pyarrow.parquet as pq

parquet_path = r"C:\Users\Arthi.R\Downloads\train-00000-of-00001.parquet"
output_dir = Path("brinjal_leaf_dataset")

label_names = {
    0: "Healthy_Leaf",
    1: "Insect_Pest_Disease",
    2: "Leaf_Spot_Disease",
    3: "Mosaic_Virus_Disease",
    4: "White_Mold_Disease",
    5: "Wilt_Disease",
}

for name in label_names.values():
    (output_dir / name).mkdir(parents=True, exist_ok=True)

parquet_file = pq.ParquetFile(parquet_path)
saved = {label: 0 for label in label_names}

for batch in parquet_file.iter_batches(batch_size=100):
    for row in batch.to_pylist():
        label = row["label"]
        if label not in label_names:
            continue

        image_data = row["image"]["bytes"]
        original_name = row["image"]["path"] or "leaf.jpg"
        safe_name = Path(original_name).name
        destination = output_dir / label_names[label] / safe_name

        # Avoid overwriting if filenames repeat
        if destination.exists():
            destination = destination.with_name(
                f"{destination.stem}_{saved[label]}{destination.suffix}"
            )

        if image_data:
            destination.write_bytes(image_data)
            saved[label] += 1

with open(output_dir / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(label_names, f, indent=2)

print("\nExtraction complete!")
for label, count in saved.items():
    print(f"{label_names[label]}: {count} images")
print(f"\nDataset folder: {output_dir.resolve()}")
