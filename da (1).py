import sys
import time
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as sql_f
import numpy as np

# Parse arguments
core = int(sys.argv[1]) if len(sys.argv) > 1 else 1
pct = int(sys.argv[2]) if len(sys.argv) > 2 else 100
filename = sys.argv[3] if len(sys.argv) > 3 else "kmeans_results"

# Start Spark session
spark = SparkSession.builder.master(f"local[{core}]").appName("KMeansExperiment").getOrCreate()
sc = spark.sparkContext

# Read CSV
df = spark.read.csv("/content/drive/MyDrive/csc735spring2025/DA_project/data/nasa.csv", header=True, inferSchema=True)

# Cast all columns to float
for col in df.columns:
    if col != "Hazardous":
        df = df.withColumn(col, df[col].cast("float"))

# Cast Hazardous column to integer
df = df.withColumn("Hazardous", df["Hazardous"].cast("int"))

# Drop rows with nulls
df = df.dropna()

# Reduce dataset percentage if specified
if pct < 100:
    df = df.sample(withReplacement=False, fraction=pct / 100.0, seed=42)

# Assemble features
features_cols = [col for col in df.columns if col != 'Hazardous']
assembler = VectorAssembler(inputCols=features_cols, outputCol='features')
df = assembler.transform(df)

# Runtime measurement
start_total = time.time()
kmeans = KMeans(k=2, seed=42, featuresCol='features', predictionCol='cluster')
model = kmeans.fit(df)
mid_total = time.time()

predictions = model.transform(df)
end_total = time.time()

# Convert predictions to Pandas for manual cluster-label mapping
preds_pd = predictions.select("Hazardous", "cluster").toPandas()
y_true = preds_pd['Hazardous'].astype(str).values
clusters = preds_pd['cluster'].values
mapped_labels = np.empty_like(clusters, dtype=object)

# Manual majority-vote mapping
for cluster_id in np.unique(clusters):
    mask = clusters == cluster_id
    labels_in_cluster = y_true[mask]
    values, counts = np.unique(labels_in_cluster, return_counts=True)
    majority_label = values[np.argmax(counts)]
    mapped_labels[mask] = majority_label

# Accuracy calculation
accuracy = np.mean(mapped_labels == y_true)

# Store results
runtime = end_total - start_total
runtime_no_overhead = end_total - mid_total

result = {
    "cores": core,
    "pct": pct,
    "accuracy": round(accuracy, 4),
    "runtime": round(runtime, 3),
    "runtime_no_overhead": round(runtime_no_overhead, 3)
}

# Save result to CSV
output_path = f"{filename}.csv"
df_result = pd.DataFrame([result])
if os.path.exists(output_path):
    df_existing = pd.read_csv(output_path)
    df_result = pd.concat([df_existing, df_result], ignore_index=True)

df_result.to_csv(output_path, index=False)
print(f" Results written to {output_path}")

# Stop Spark
spark.stop()
