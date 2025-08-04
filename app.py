from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pyreadstat # New library for reading .sav files
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import io # Required to handle the file stream

app = Flask(__name__)
# Allow requests from any origin for development. 
# For production, you might want to restrict this to your frontend's URL.
CORS(app) 

@app.route('/api/upload', methods=['POST'])
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
            # Read the file content into a BytesIO object to be used by libraries
            file_content = io.BytesIO(file.read())
            
            if file.filename.endswith('.sav'):
                # Use pyreadstat for .sav files
                df, meta = pyreadstat.read_sav(file_content)
                
                # Extract relevant metadata
                metadata = {
                    "variable_labels": meta.column_names_to_labels,
                    "value_labels": meta.variable_value_labels,
                    "missing_ranges": meta.missing_ranges,
                }
            else: # .csv file
                # Use pandas for .csv files
                df = pd.read_csv(file_content)
                metadata = {} # CSV files have no standard metadata

            # Convert dataframe to JSON records format
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
        # Provide a more specific error message if possible
        return jsonify({"error": f"An error occurred while processing the file: {str(e)}"}), 500


@app.route('/api/crosstab', methods=['POST'])
def perform_crosstab():
    """
    Performs cross-tabulation on the provided data.
    Expects a JSON payload with 'data', 'var1', and 'var2'.
    """
    try:
        payload = request.get_json()
        data = payload.get('data')
        var1 = payload.get('var1')
        var2 = payload.get('var2')

        if not all([data, var1, var2]):
            return jsonify({"error": "Missing data, var1, or var2 in request."}), 400

        df = pd.DataFrame(data)
        
        # Create the cross-tabulation table
        crosstab_table = pd.crosstab(df[var1], df[var2])
        
        # Convert to a format that can be sent as JSON
        crosstab_json = json.loads(crosstab_table.to_json(orient='split'))

        return jsonify({
            "message": "Cross-tabulation successful!",
            "table": crosstab_json
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/kmeans', methods=['POST'])
def perform_kmeans():
    """
    Performs K-Means clustering on the provided data.
    Expects a JSON payload with 'data', 'k', and 'columns'.
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
        
        # Select only the specified columns for clustering and handle non-numeric data
        cluster_data = df[columns].copy()
        
        # Convert all data to numeric, coercing errors
        for col in cluster_data.columns:
            cluster_data[col] = pd.to_numeric(cluster_data[col], errors='coerce')
        
        # Drop rows with missing values that might result from coercion
        cluster_data.dropna(inplace=True)

        if cluster_data.shape[0] < k:
             return jsonify({"error": "Not enough data points to form the requested number of clusters after cleaning."}), 400

        # Scale the data for better clustering performance
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        
        # Add the cluster labels back to the original dataframe (aligning by index)
        df['Cluster'] = pd.Series(kmeans.labels_, index=cluster_data.index)
        # Fill non-clustered rows with a placeholder
        df['Cluster'].fillna('N/A', inplace=True)


        # Calculate cluster summaries
        summary = df.groupby('Cluster').size().reset_index(name='Count')

        return jsonify({
            "message": "K-Means clustering successful!",
            "dataWithClusters": json.loads(df.to_json(orient='records')),
            "summary": json.loads(summary.to_json(orient='records'))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Runs the app on http://127.0.0.1:5000
    app.run(debug=True)
