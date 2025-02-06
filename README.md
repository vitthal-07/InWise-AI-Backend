# Smart Invoice Matching

## Contributors

- **Anup Nalwade**
- **Vitthal Biradar**
- **Bharat Patil**
- **Sumit Tippanbone**

---

Smart Invoice Matching is a Flask-based web service that processes purchase orders and invoices. It leverages machine learning techniques—including TensorFlow and scikit-learn—to perform matching and generate reports. The application also uses Cloudinary for file storage and MongoDB for record keeping.

## Features 

- **File Processing:** Accepts CSV or Excel files for purchase orders and invoices.
- **Data Matching:** Uses TF-IDF vectorization, cosine similarity, and additional heuristics (SequenceMatcher) to match invoice records.
- **Machine Learning:** Trains a neural network model (using TensorFlow) to evaluate and classify matches.
- **Dimensionality Reduction:** Implements dynamic dimensionality reduction with TruncatedSVD.
- **Report Generation:** Generates matching reports in Excel format and uploads them to Cloudinary.
- **API Endpoints:**
  - `/process`: Processes PO and Invoice files, performs matching, trains the model, and stores report data.
  - `/manual_match`: Allows manual match corrections.
  - `/fetch_entries`: Retrieves stored matching entries from MongoDB.
  - `/process_xlsx`: Processes and filters Excel reports with pagination.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/smart-invoice-matching.git
   cd smart-invoice-matching
