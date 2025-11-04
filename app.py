"""
Production-Grade Intent Classification API
Multi-model support with A/B testing and monitoring
"""

from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import json
import sqlite3
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
models = {}
vectorizer = None
label_map = {}
reverse_map = {}
model_metadata = {}

def init_database():
    conn = sqlite3.connect('data/logs.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            intent TEXT,
            confidence FLOAT,
            model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("✓ Database initialized")

def load_models():
    global models, vectorizer, label_map, reverse_map, model_metadata
    
    try:
        current_file = Path('models/current')
        with open(current_file, 'r') as f:
            version = f.read().strip()
        
        version_dir = Path('models') / version
        
        # Load vectorizer
        with open(version_dir / 'vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load label map
        with open(version_dir / 'label_map.json', 'r') as f:
            label_map = json.load(f)
        
        reverse_map = {v: k for k, v in label_map.items()}
        
        # Load models
        for model_file in version_dir.glob('*.pkl'):
            if model_file.name != 'vectorizer.pkl':
                name = model_file.stem
                with open(model_file, 'rb') as f:
                    models[name] = pickle.load(f)
                logger.info(f"   ✓ Loaded {name}")
        
        # Load metadata
        results_file = version_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                model_metadata = json.load(f)
        
        logger.info(f"✓ Loaded {len(models)} models")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

def predict_intent(text, model_name='random_forest'):
    if model_name not in models:
        model_name = list(models.keys())[0]
    
    text_vec = vectorizer.transform([text])
    model = models[model_name]
    
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    intent = reverse_map[prediction]
    confidence = float(probabilities[prediction])
    
    prob_dict = {reverse_map[i]: float(p) for i, p in enumerate(probabilities)}
    
    return {
        'intent': intent,
        'confidence': confidence,
        'probabilities': prob_dict,
        'model_used': model_name
    }

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models': list(models.keys()),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        query = data.get('query')
        model_name = data.get('model', 'random_forest')
        
        result = predict_intent(query, model_name)
        result['query'] = query
        result['timestamp'] = datetime.now().isoformat()
        
        # Log to database
        conn = sqlite3.connect('data/logs.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO predictions (query, intent, confidence, model_used)
            VALUES (?, ?, ?, ?)
        ''', (query, result['intent'], result['confidence'], model_name))
        conn.commit()
        conn.close()
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data = request.get_json()
        query = data.get('query')
        
        results = {}
        for model_name in models.keys():
            results[model_name] = predict_intent(query, model_name)
        
        return jsonify({
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    try:
        conn = sqlite3.connect('data/logs.db')
        c = conn.cursor()
        
        c.execute('SELECT COUNT(*) FROM predictions')
        total = c.fetchone()[0]
        
        c.execute('SELECT intent, COUNT(*) FROM predictions GROUP BY intent')
        intent_dist = dict(c.fetchall())
        
        c.execute('SELECT AVG(confidence) FROM predictions')
        avg_conf = c.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'total_predictions': total,
            'intent_distribution': intent_dist,
            'avg_confidence': avg_conf,
            'models': model_metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("CUSTOMER INTENT CLASSIFICATION API")
    print("=" * 70)
    print()
    
    init_database()
    
    if not load_models():
        print("❌ Failed to load models")
        exit(1)
    
    print()
    print("=" * 70)
    print("API Endpoints:")
    print("  POST /api/predict  - Predict intent")
    print("  POST /api/compare  - Compare models")
    print("  GET  /api/stats    - Get statistics")
    print("=" * 70)
    print()
    print("Starting server on http://localhost:5000")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
