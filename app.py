"""
PERM EB2 Approval Prediction - Flask Web Application
=====================================================
This app loads the best model (selected by F1 Score) and provides
a web interface for predicting PERM EB2 approval probability.

To run locally:
    pip install -r requirements.txt
    python app.py

To deploy on platforms like Render, Heroku, or Railway:
    1. Push this code to GitHub
    2. Connect your repo to the platform
    3. Set build command: pip install -r requirements.txt
    4. Set start command: gunicorn app:app
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# =================================================================================
# LOAD MODEL
# =================================================================================
# The model was trained and saved in Colab using F1 Score as the primary metric
# Make sure 'best_model_for_deployment.pkl' is in the same directory as app.py

MODEL_PATH = 'best_model_for_deployment.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    print("Please ensure the model file is in the same directory as app.py")
    model = None


# =================================================================================
# FEATURE CONFIGURATION
# =================================================================================
# These must match the features used during training

NUMERIC_FEATURES = [
    "PW_WAGE_ANNUAL", "OFFER_WAGE_ANNUAL", "WAGE_RATIO", "WAGE_PREMIUM",
    "LOG_PW_WAGE", "LOG_OFFER_WAGE", "EDUCATION_LEVEL", "HAS_OWNERSHIP", "DATA_YEAR"
]

CATEGORICAL_FEATURES = ["SOC_MAJOR", "NAICS_SECTOR", "REGION", "WORKSITE_STATE"]

# Education level mapping
EDUCATION_MAP = {
    "none": 0,
    "high_school": 1,
    "associates": 2,
    "bachelors": 3,
    "masters": 4,
    "doctorate": 5,
    "other": 2
}

# State to region mapping
STATE_REGIONS = {
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
    'ND': 'Midwest', 'SD': 'Midwest',
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South',
    'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
    'OK': 'South', 'TX': 'South',
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West'
}


# =================================================================================
# HELPER FUNCTIONS
# =================================================================================

def preprocess_input(form_data):
    """
    Convert form input to the format expected by the model.
    """
    # Parse numeric values
    pw_wage = float(form_data.get('pw_wage', 0))
    offer_wage_from = float(form_data.get('offer_wage_from', 0))
    offer_wage_to = float(form_data.get('offer_wage_to', offer_wage_from))
    
    # Calculate wage features
    offer_wage = (offer_wage_from + offer_wage_to) / 2  # Midpoint
    wage_ratio = offer_wage / pw_wage if pw_wage > 0 else 1
    wage_premium = offer_wage - pw_wage
    
    # Log-transformed wages
    log_pw_wage = np.log1p(pw_wage)
    log_offer_wage = np.log1p(offer_wage)
    
    # Education level
    education = form_data.get('education', 'bachelors').lower()
    education_level = EDUCATION_MAP.get(education, 2)
    
    # Ownership interest
    has_ownership = 1 if form_data.get('ownership', 'N') == 'Y' else 0
    
    # Data year (use current or specified)
    data_year = int(form_data.get('year', 2024))
    
    # State and region
    state = form_data.get('state', 'CA').upper()
    region = STATE_REGIONS.get(state, 'West')
    
    # SOC code (first 2 digits = major group)
    soc_code = form_data.get('soc_code', '15-0000')
    soc_major = str(soc_code)[:2]
    
    # NAICS code (first 2 digits = sector)
    naics_code = form_data.get('naics_code', '54')
    naics_sector = str(naics_code)[:2]
    
    # Create DataFrame with exact column names used in training
    input_data = pd.DataFrame([{
        'PW_WAGE_ANNUAL': pw_wage,
        'OFFER_WAGE_ANNUAL': offer_wage,
        'WAGE_RATIO': wage_ratio,
        'WAGE_PREMIUM': wage_premium,
        'LOG_PW_WAGE': log_pw_wage,
        'LOG_OFFER_WAGE': log_offer_wage,
        'EDUCATION_LEVEL': education_level,
        'HAS_OWNERSHIP': has_ownership,
        'DATA_YEAR': data_year,
        'SOC_MAJOR': soc_major,
        'NAICS_SECTOR': naics_sector,
        'REGION': region,
        'WORKSITE_STATE': state
    }])
    
    return input_data


# =================================================================================
# ROUTES
# =================================================================================

@app.route('/')
def home():
    """Render the main prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    Accepts both form submissions and JSON API calls.
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure best_model_for_deployment.pkl exists.'
        }), 500
    
    try:
        # Get input data (from form or JSON)
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()
        
        # Preprocess input
        input_df = preprocess_input(form_data)
        
        # Get prediction probability
        proba = model.predict_proba(input_df)[0]
        approval_probability = proba[1]  # Probability of approval (class 1)
        
        # Get binary prediction
        prediction = model.predict(input_df)[0]
        
        # Determine risk level
        if approval_probability >= 0.8:
            risk_level = "LOW RISK"
            risk_color = "green"
            recommendation = "Strong case. Standard processing recommended."
        elif approval_probability >= 0.6:
            risk_level = "MODERATE RISK"
            risk_color = "orange"
            recommendation = "Review case details. Consider strengthening weak areas."
        else:
            risk_level = "HIGH RISK"
            risk_color = "red"
            recommendation = "Significant concerns. Detailed review recommended before filing."
        
        result = {
            'approval_probability': round(approval_probability * 100, 2),
            'prediction': 'APPROVED' if prediction == 1 else 'DENIED',
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'input_summary': {
                'pw_wage': form_data.get('pw_wage'),
                'offer_wage': f"{form_data.get('offer_wage_from')} - {form_data.get('offer_wage_to', form_data.get('offer_wage_from'))}",
                'education': form_data.get('education'),
                'state': form_data.get('state'),
                'soc_code': form_data.get('soc_code'),
                'naics_code': form_data.get('naics_code')
            }
        }
        
        if request.is_json:
            return jsonify(result)
        else:
            return render_template('result.html', result=result)
    
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        if request.is_json:
            return jsonify({'error': error_msg}), 400
        else:
            return render_template('error.html', error=error_msg), 400


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for programmatic access.
    Accepts JSON input and returns JSON response.
    
    Example request:
    curl -X POST http://localhost:5000/api/predict \
         -H "Content-Type: application/json" \
         -d '{"pw_wage": 100000, "offer_wage_from": 120000, "education": "masters", "state": "CA", "soc_code": "15-1252", "naics_code": "54"}'
    """
    return predict()


@app.route('/health')
def health():
    """Health check endpoint for deployment platforms."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


# =================================================================================
# HTML TEMPLATES (Embedded for simplicity)
# =================================================================================
# In production, move these to a 'templates' folder

HTML_INDEX = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PERM EB2 Approval Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .row {
            display: flex;
            gap: 15px;
        }
        .row .form-group {
            flex: 1;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }
        .info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 13px;
            color: #666;
        }
        .metric-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏛️ PERM EB2 Predictor</h1>
        <p class="subtitle">Predict approval probability for PERM labor certification</p>
        <span class="metric-badge">Model optimized for F1 Score</span>
        
        <form action="/predict" method="POST" style="margin-top: 30px;">
            <div class="row">
                <div class="form-group">
                    <label>Prevailing Wage (Annual $)</label>
                    <input type="number" name="pw_wage" placeholder="e.g., 100000" required>
                </div>
            </div>
            
            <div class="row">
                <div class="form-group">
                    <label>Offered Wage From ($)</label>
                    <input type="number" name="offer_wage_from" placeholder="e.g., 120000" required>
                </div>
                <div class="form-group">
                    <label>Offered Wage To ($)</label>
                    <input type="number" name="offer_wage_to" placeholder="e.g., 150000">
                </div>
            </div>
            
            <div class="form-group">
                <label>Minimum Education Required</label>
                <select name="education">
                    <option value="bachelors">Bachelor's Degree</option>
                    <option value="masters">Master's Degree</option>
                    <option value="doctorate">Doctorate</option>
                    <option value="associates">Associate's Degree</option>
                    <option value="high_school">High School</option>
                    <option value="none">None</option>
                </select>
            </div>
            
            <div class="row">
                <div class="form-group">
                    <label>Worksite State</label>
                    <select name="state">
                        <option value="CA">California</option>
                        <option value="TX">Texas</option>
                        <option value="NY">New York</option>
                        <option value="WA">Washington</option>
                        <option value="NJ">New Jersey</option>
                        <option value="IL">Illinois</option>
                        <option value="MA">Massachusetts</option>
                        <option value="FL">Florida</option>
                        <option value="PA">Pennsylvania</option>
                        <option value="GA">Georgia</option>
                        <option value="VA">Virginia</option>
                        <option value="NC">North Carolina</option>
                        <option value="OH">Ohio</option>
                        <option value="MI">Michigan</option>
                        <option value="AZ">Arizona</option>
                        <option value="CO">Colorado</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Filing Year</label>
                    <select name="year">
                        <option value="2024">2024</option>
                        <option value="2025">2025</option>
                    </select>
                </div>
            </div>
            
            <div class="row">
                <div class="form-group">
                    <label>SOC Code (Job Classification)</label>
                    <input type="text" name="soc_code" placeholder="e.g., 15-1252" value="15-1252">
                </div>
                <div class="form-group">
                    <label>NAICS Code (Industry)</label>
                    <input type="text" name="naics_code" placeholder="e.g., 54" value="54">
                </div>
            </div>
            
            <div class="form-group">
                <label>Foreign Worker Has Ownership Interest?</label>
                <select name="ownership">
                    <option value="N">No</option>
                    <option value="Y">Yes</option>
                </select>
            </div>
            
            <button type="submit">🔮 Predict Approval Probability</button>
        </form>
        
        <div class="info">
            <strong>Note:</strong> This prediction is based on historical PERM data (2022-2024) 
            and should be used as one factor in case assessment. Model selected using F1 Score 
            to optimize for accurate approval predictions.
        </div>
    </div>
</body>
</html>
"""

HTML_RESULT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result - PERM EB2</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
            text-align: center;
        }
        h1 { color: #333; margin-bottom: 30px; }
        .probability-circle {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            margin: 0 auto 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            font-size: 48px;
            font-weight: bold;
            color: white;
        }
        .probability-circle.green { background: linear-gradient(135deg, #00b894, #00cec9); }
        .probability-circle.orange { background: linear-gradient(135deg, #fdcb6e, #e17055); }
        .probability-circle.red { background: linear-gradient(135deg, #e74c3c, #c0392b); }
        .probability-circle span { font-size: 14px; font-weight: normal; }
        .risk-badge {
            display: inline-block;
            padding: 10px 25px;
            border-radius: 25px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .risk-badge.green { background: #d4edda; color: #155724; }
        .risk-badge.orange { background: #fff3cd; color: #856404; }
        .risk-badge.red { background: #f8d7da; color: #721c24; }
        .recommendation {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            color: #555;
        }
        .details {
            text-align: left;
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .details h3 { margin-bottom: 15px; color: #333; }
        .details p { margin: 5px 0; color: #666; font-size: 14px; }
        .back-btn {
            display: inline-block;
            margin-top: 30px;
            padding: 15px 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
        }
        .back-btn:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 Prediction Result</h1>
        
        <div class="probability-circle {{ result.risk_color }}">
            {{ result.approval_probability }}%
            <span>Approval Probability</span>
        </div>
        
        <div class="risk-badge {{ result.risk_color }}">
            {{ result.risk_level }}
        </div>
        
        <div class="recommendation">
            <strong>Recommendation:</strong><br>
            {{ result.recommendation }}
        </div>
        
        <div class="details">
            <h3>📋 Input Summary</h3>
            <p><strong>Prevailing Wage:</strong> ${{ result.input_summary.pw_wage }}</p>
            <p><strong>Offered Wage:</strong> ${{ result.input_summary.offer_wage }}</p>
            <p><strong>Education:</strong> {{ result.input_summary.education }}</p>
            <p><strong>State:</strong> {{ result.input_summary.state }}</p>
            <p><strong>SOC Code:</strong> {{ result.input_summary.soc_code }}</p>
            <p><strong>NAICS Code:</strong> {{ result.input_summary.naics_code }}</p>
        </div>
        
        <a href="/" class="back-btn">← Make Another Prediction</a>
    </div>
</body>
</html>
"""

HTML_ERROR = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - PERM EB2 Predictor</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 16px;
            text-align: center;
            max-width: 500px;
        }
        h1 { color: #e74c3c; margin-bottom: 20px; }
        p { color: #666; margin-bottom: 20px; }
        a {
            display: inline-block;
            padding: 15px 40px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚠️ Error</h1>
        <p>{{ error }}</p>
        <a href="/">← Go Back</a>
    </div>
</body>
</html>
"""


# =================================================================================
# TEMPLATE RENDERING OVERRIDE (for embedded templates)
# =================================================================================

from flask import Response
from jinja2 import Template

_original_render_template = render_template

def render_template(template_name, **context):
    """Override to use embedded templates."""
    templates = {
        'index.html': HTML_INDEX,
        'result.html': HTML_RESULT,
        'error.html': HTML_ERROR
    }
    
    if template_name in templates:
        template = Template(templates[template_name])
        return Response(template.render(**context), mimetype='text/html')
    
    return _original_render_template(template_name, **context)


# =================================================================================
# MAIN
# =================================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          PERM EB2 Approval Prediction Service                ║
    ║                                                              ║
    ║  Model optimized for: F1 SCORE                               ║
    ║  Running on: http://localhost:{port}                          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host='0.0.0.0', port=port, debug=debug)
