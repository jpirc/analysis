from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pyreadstat
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import io

app = Flask(__name__)
CORS(app) 

# The path is now just /upload
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles the upload of a .sav or .csv file.
    Reads the file and returns the data and metadata as JSON.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and (file.filename.endswith('.sav') or file.filename.endswith('.csv')):
            file_content = io.BytesIO(file.read())
            
            if file.filename.endswith('.sav'):
                df, meta = pyreadstat.read_sav(file_content)
                metadata = {
                    "variable_labels": meta.column_names_to_labels,
                    "value_labels": meta.variable_value_labels,
                    "missing_ranges": meta.missing_ranges,
                }
            else:
                df = pd.read_csv(file_content)
                metadata = {}

            data_json = json.loads(df.to_json(orient='records'))

            return jsonify({
                "message": "File processed successfully!",
                "data": data_json,
                "headers": list(df.columns),
                "metadata": metadata
            })
        else:
            return jsonify({"error": "Invalid file type. Please upload a .sav or .csv file."}), 400

    except Exception as e:
        return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500

# The path is now just /crosstab
@app.route('/crosstab', methods=['POST'])
def perform_crosstab():
    """
    Performs cross-tabulation on the provided data.
    """
    try:
        payload = request.get_json()
        data = payload.get('data')
        var1 = payload.get('var1')
        var2 = payload.get('var2')

        if not all([data, var1, var2]):
            return jsonify({"error": "Missing data, var1, or var2 in request."}), 400

        df = pd.DataFrame(data)
        crosstab_table = pd.crosstab(df[var1], df[var2])
        crosstab_json = json.loads(crosstab_table.to_json(orient='split'))

        return jsonify({
            "message": "Cross-tabulation successful!",
            "table": crosstab_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# The path is now just /kmeans
@app.route('/kmeans', methods=['POST'])
def perform_kmeans():
    """
    Performs K-Means clustering on the provided data.
    """
    try:
        payload = request.get_json()
        data = payload.get('data')
        k = int(payload.get('k', 3))
        columns = payload.get('columns')

        if not all([data, k, columns]):
            return jsonify({"error": "Missing data, k, or columns in request."}), 400
        
        if len(columns) == 0:
            return jsonify({"error": "Please select at least one column for clustering."}), 400

        df = pd.DataFrame(data)
        cluster_data = df[columns].copy()
        
        for col in cluster_data.columns:
            cluster_data[col] = pd.to_numeric(cluster_data[col], errors='coerce')
        
        cluster_data.dropna(inplace=True)

        if cluster_data.shape[0] < k:
             return jsonify({"error": "Not enough data points to form the requested number of clusters after cleaning."}), 400

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        df['Cluster'] = pd.Series(kmeans.labels_, index=cluster_data.index)
        df['Cluster'].fillna('N/A', inplace=True)

        summary = df.groupby('Cluster').size().reset_index(name='Count')

        return jsonify({
            "message": "K-Means clustering successful!",
            "dataWithClusters": json.loads(df.to_json(orient='records')),
            "summary": json.loads(summary.to_json(orient='records'))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
