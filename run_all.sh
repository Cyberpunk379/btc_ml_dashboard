#!/bin/bash

echo "📥 Running Ingestion..."
python ingest_data.py || exit 1

echo "🧪 Running Feature Engineering..."
python feature_engineering.py || exit 1

echo "📈 Running Model Serving..."
python pipeline_serve.py || exit 1

echo "✅ Full pipeline completed."
