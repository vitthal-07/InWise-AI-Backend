import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_pymongo import PyMongo
from datetime import datetime
import pandas as pd
import numpy as np
import cloudinary
import cloudinary.uploader
import cloudinary.api
from io import BytesIO
import requests
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from difflib import SequenceMatcher
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from pymongo import MongoClient
from bson import ObjectId
import re

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure Cloudinary using environment variables
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

app = Flask(__name__)
CORS(app)

# MongoDB Configuration from environment variables
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo = PyMongo(app)

# Constants
COLLECTION_NAME = 'matches'
MODEL_PATH = os.getenv('MODEL_PATH', "modal/matcher_model2.h5")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ---------------------------
# Helper Functions
# ---------------------------

def load_input_file(file_storage):
    """Load CSV or Excel file into DataFrame"""
    try:
        filename = file_storage.filename.lower()
        if filename.endswith('.csv'):
            return pd.read_csv(file_storage)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_storage)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise RuntimeError(f"Error loading file: {str(e)}")

def create_new_model(input_dim):
    """Create a new neural network model"""
    input_layer = Input(shape=(input_dim,))
    dense_layer = Dense(128, activation='relu')(input_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(0.001), 
                 loss='binary_crossentropy', 
                 metrics=['accuracy'])
    return model

def process_matching(po_df, invoice_df, po_column, invoice_column):
    """Main matching logic with dynamic columns"""
    try:
        # Preprocessing
        po_df[po_column] = po_df[po_column].astype(str).str.strip().str.lower()
        invoice_df[invoice_column] = invoice_df[invoice_column].astype(str).str.strip().str.lower()

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf_po = vectorizer.fit_transform(po_df[po_column]).astype(np.float32)
        tfidf_invoice = vectorizer.transform(invoice_df[invoice_column]).astype(np.float32)

        matches = []
        for idx in range(tfidf_po.shape[0]):
            similarities = cosine_similarity(tfidf_po[idx:idx+1], tfidf_invoice).flatten()
            best_match_idx = np.argmax(similarities)
            confidence = similarities[best_match_idx] * 100
            po_item = po_df.iloc[idx][po_column]
            invoice_item = invoice_df.iloc[best_match_idx][invoice_column] if confidence > 0 else ""

            # Match classification
            if po_item == invoice_item:
                status, confidence = "exact", 100.0
            elif confidence == 0:
                status = "unmatched"
            else:
                seq_ratio = SequenceMatcher(None, po_item, invoice_item).ratio() * 100
                if any(x in y for x, y in [(po_item, invoice_item), (invoice_item, po_item)]):
                    status, confidence = "partial", seq_ratio
                else:
                    status, confidence = "unmatched", min(seq_ratio, 50.0)

            matches.append({
                'match_id': idx + 1,
                'invoice_1': po_item,
                'invoice_2': invoice_item,
                'status': status,
                'confidence_score': confidence
            })

        return pd.DataFrame(matches), vectorizer

    except Exception as e:
        raise RuntimeError(f"Matching error: {str(e)}")

def reduce_dimensions(tfidf_matrix):
    """Dynamic dimensionality reduction"""
    n_features = tfidf_matrix.shape[1]
    n_components = max(1, min(500, n_features))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(tfidf_matrix), svd

# ---------------------------
# Flask Routes
# ---------------------------

@app.route('/process', methods=['POST'])
def process_files():
    try:
        # File validation
        if 'po_file' not in request.files or 'invoice_file' not in request.files:
            return jsonify({"error": "Both PO and Invoice files required"}), 400

        po_file = request.files['po_file']
        invoice_file = request.files['invoice_file']

        # Load data
        po_df = load_input_file(po_file)
        invoice_df = load_input_file(invoice_file)

        # Column validation
        if po_df.empty or invoice_df.empty:
            return jsonify({"error": "Files cannot be empty"}), 400
        po_col = po_df.columns[0]
        invoice_col = invoice_df.columns[0]

        # Process matching
        report_df, vectorizer = process_matching(po_df, invoice_df, po_col, invoice_col)

        # Dimensionality reduction
        tfidf_matrix = vectorizer.transform(po_df[po_col]).astype(np.float32)
        tfidf_reduced, _ = reduce_dimensions(tfidf_matrix)

        # Model handling
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            if model.input_shape[1] != tfidf_reduced.shape[1]:
                model = create_new_model(tfidf_reduced.shape[1])
        else:
            model = create_new_model(tfidf_reduced.shape[1])

        # Model training
        y = np.array([1 if s == "exact" else 0 for s in report_df['status']])
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf_reduced, y, test_size=0.2, random_state=42
        )
        # Model evaluation
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracy_value = float(accuracy)  # Convert to Python float
        model.save(MODEL_PATH)

        # Generate report
        output = io.BytesIO()
        report_df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)

        # Cloudinary upload
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        upload_result = cloudinary.uploader.upload(
            output,
            resource_type="raw",
            public_id=f"matching_report_{timestamp}.xlsx",
        )

        # MongoDB storage
        result = mongo.db[COLLECTION_NAME].insert_one({
            "file_url": upload_result['secure_url'],
            "timestamp": datetime.utcnow(),
            "accuracy": accuracy_value
        })

        new_document = mongo.db[COLLECTION_NAME].find_one({"_id": result.inserted_id})

        return jsonify({
            "data": new_document
        })

    except Exception as e:
        app.logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def extract_public_id(url):
    """Extract public ID from Cloudinary URL"""
    try:
        parts = url.split('/upload/')
        after_upload = parts[1].split('/')
        return '/'.join(after_upload[1:])  # Skip version part
    except Exception:
        return None
    
@app.route('/manual_match', methods=['POST'])
def manual_match():
    try:
        data = request.get_json()
        invoice_1 = data.get('invoice_1')
        invoice_2 = data.get('invoice_2')
        cloudinary_url = data.get('cloudinary_url')
        match_id = data.get('match_id')

        if not all([invoice_1, invoice_2, cloudinary_url, match_id]):
            return jsonify({"error": "Missing required fields"}), 400

        # Download Excel file
        response = requests.get(cloudinary_url)

        # Read Excel file
        df = pd.read_excel(BytesIO(response.content))

        # Validate columns
        required_columns = ['match_id', 'invoice_1', 'invoice_2', 'status', 'confidence_score']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Invalid file format"}), 400

        # Create new entry
        if df['match_id'].dtype == 'int64' and not df.empty:
            last_match_id = df['match_id'].max()
            new_match_id = last_match_id + 1
        else:
            new_match_id = 1  # Start from 1 if the DataFrame is empty or match_id is not integer

        new_entry = {
            'match_id': new_match_id,
            'invoice_1': invoice_1,
            'invoice_2': invoice_2,
            'status': 'manual match',
            'confidence_score': 100
        }

        # Add to DataFrame
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)

        # Prepare updated file
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)

        # Upload to Cloudinary
        public_id = extract_public_id(cloudinary_url)
        if not public_id:
            return jsonify({"error": "Invalid Cloudinary URL"}), 400
        print(public_id)
        upload_result = cloudinary.uploader.upload(
            output,
            public_id=public_id,
            resource_type='raw',
            overwrite=True
        )
        db_entry = {
            'file_url': upload_result['secure_url'],
            'timestamp': datetime.utcnow()
        }

        mongo.db[COLLECTION_NAME].update_one(
            {'_id': match_id},
            {'$set': db_entry},
            upsert=False
        )

        return jsonify({
            "message": "Match added successfully",
            "url": upload_result['secure_url']
        }), 200

    except requests.RequestException as e:
        return jsonify({"error": f"File download failed: {str(e)}"}), 400
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Empty Excel file"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fetch_entries', methods=['GET'])
def fetch_entries():
    try:
        # Fetch data from MongoDB
        matches = mongo.db[COLLECTION_NAME].find()
        result = [match for match in matches]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def preprocess_data(data):
    for entry in data:
        if isinstance(entry.get("invoice_2"), dict) and "$numberDouble" in entry["invoice_2"]:
            entry["invoice_2"] = "N/A"
    return data

@app.route('/process_xlsx', methods=['POST'])
def process_xlsx():
    try:
        data = request.get_json()
        cloudinary_url = data.get('cloudinary_url')
        if not cloudinary_url:
            return jsonify({"error": "Cloudinary URL is required"}), 400

        # Get query parameters for pagination and filtering
        page = request.args.get('page', default=1, type=int)
        rows_per_page = 100

        # Filtering parameters
        date_start_str = request.args.get('dateStart', default="", type=str)
        date_end_str = request.args.get('dateEnd', default="", type=str)
        min_confidence = request.args.get('minConfidence', default=0, type=float)
        max_confidence = request.args.get('maxConfidence', default=100, type=float)
        status_str = request.args.get('status', default="", type=str)
        search_term = request.args.get('searchTerm', default="", type=str)

        # Convert comma-separated status string into a list (if provided)
        status_filter = [s.strip() for s in status_str.split(',')] if status_str else []

        # Retrieve file from Cloudinary
        response = requests.get(cloudinary_url)
        file_content = BytesIO(response.content)

        # Read Excel file
        df = pd.read_excel(file_content, engine='openpyxl')

        # Check if required columns exist
        required_columns = ['match_id', 'invoice_1', 'invoice_2', 'status', 'confidence_score']
        if not all(col in df.columns for col in required_columns):
            return jsonify({"error": "Missing required columns in the XLSX file"}), 400

        # Ensure invoice_2 is string and handle NaN values
        df['invoice_2'] = df['invoice_2'].astype(str).replace('nan', 'N/A')

        # -------------------------------
        # Apply Filters
        # -------------------------------

        # Filter by confidence score
        df = df[(df['confidence_score'] >= min_confidence) & (df['confidence_score'] <= max_confidence)]

        # Filter by status if provided
        if status_filter:
            df = df[df['status'].isin(status_filter)]

        # Filter by search term (in invoice_1 or invoice_2)
        if search_term:
            search_term_lower = search_term.lower()
            df = df[
                df['invoice_1'].str.lower().str.contains(search_term_lower) |
                df['invoice_2'].str.lower().str.contains(search_term_lower)
            ]

        # If date filtering is needed, assume the file has a column "match_date"
        # and convert it to datetime. Adjust the column name as needed.
        if date_start_str and date_end_str and 'match_date' in df.columns:
            try:
                df['match_date'] = pd.to_datetime(df['match_date'])
                date_start = pd.to_datetime(date_start_str)
                date_end = pd.to_datetime(date_end_str)
                df = df[(df['match_date'] >= date_start) & (df['match_date'] <= date_end)]
            except Exception as e:
                return jsonify({"error": f"Error processing date filters: {str(e)}"}), 400

        # -------------------------------
        # Pagination
        # -------------------------------
        total_rows = len(df)
        total_pages = (total_rows // rows_per_page) + (1 if total_rows % rows_per_page > 0 else 0)
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        paginated_df = df.iloc[start_idx:end_idx]

        result = paginated_df.to_dict(orient='records')

        return jsonify({
            "page": page,
            "rows_per_page": rows_per_page,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "data": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
