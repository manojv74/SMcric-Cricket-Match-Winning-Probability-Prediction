import requests
import random
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import csv
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# Add the IPLWinPredictor class definition here
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class IPLWinPredictor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.known_teams = None
        self.known_toss_decisions = None
        
    def fit_encoders(self, df):
        """Fit the label encoders on the full dataset"""
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        
        self.le_team.fit(self.known_teams)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        """Safely encode values, handling unknown categories"""
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        return encoder.transform(series)
    
    def prepare_features(self, df, is_training=True):
        """Prepare features for the model without venue information"""
        if is_training:
            self.fit_encoders(df)
        
        df_encoded = df.copy()
        
        try:
            # Basic encoding
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)
            
            # Feature engineering
            df_encoded['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
            df_encoded['is_batting_first'] = (df['toss_decision'] == 'bat').astype(int)
            
            # Create batting/bowling order features
            df_encoded['team1_batting_first'] = ((df['toss_winner'] == df['team1']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            df_encoded['team2_batting_first'] = ((df['toss_winner'] == df['team2']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            
            features = [
                'team1_encoded', 'team2_encoded',
                'toss_winner_encoded', 'toss_decision_encoded',
                'is_toss_winner_team1', 'is_batting_first',
                'team1_batting_first', 'team2_batting_first'
            ]
            
            if 'target_runs' in df.columns:
                features.extend(['target_runs', 'target_overs'])
            
            X = df_encoded[features]
            
            if is_training:
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            if is_training:
                return X_scaled, df_encoded['target']
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            print("Input data:")
            print(df.head())
            raise
    
    def train(self, df):
        """Train the model"""
        try:
            X, y = self.prepare_features(df, is_training=True)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(
                ['team1', 'team2', 'toss_winner', 'toss_decision',
                 'is_toss_winner_team1', 'is_batting_first', 
                 'team1_batting_first', 'team2_batting_first', 
                 'target_runs', 'target_overs'],
                self.model.feature_importances_
            ))
            
            return accuracy, report, feature_importance
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            raise
    
    def predict_win_probability(self, match_info):
        """Predict win probability for a match"""
        try:
            match_data = pd.DataFrame([match_info])
            X = self.prepare_features(match_data, is_training=False)
            probabilities = self.model.predict_proba(X)[0]
            
            # Adjust probabilities based on match situation
            if all(k in match_info for k in ['required_runs', 'remaining_overs', 'required_wickets']):
                required_rr = match_info['required_runs'] / max(match_info['remaining_overs'], 0.1)
                initial_rr = match_info['target_runs'] / match_info['target_overs']
                
                # Enhanced situation analysis
                rr_factor = np.clip(initial_rr / max(required_rr, 0.1), 0.5, 2.0)
                wickets_factor = match_info['required_wickets'] / 10.0
                
                # Consider batting/bowling order
                batting_first = match_info['toss_decision'] == 'bat'
                batting_first_factor = 1.1 if batting_first else 0.9
                
                # Combined situation factor
                situation_factor = rr_factor * wickets_factor * batting_first_factor
                adjusted_prob = probabilities * situation_factor
                
                # Normalize probabilities
                total_prob = np.sum(adjusted_prob)
                if total_prob > 0:
                    adjusted_prob = adjusted_prob / total_prob
                probabilities = adjusted_prob
            
            return {
                'team1_win_probability': probabilities[1],
                'team2_win_probability': probabilities[0]
            }
            
        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            raise

def main(match_info):
    # Load your cleaned IPL dataset
    df = pd.read_csv('/Users/samuel/codemfl/SMcric/output2.csv')
    
    # Initialize and train the predictor
    predictor = IPLWinPredictor()
    accuracy, report, feature_importance = predictor.train(df)
    
    #print(f"Model Accuracy: {accuracy:.2f}")
   # print("\nClassification Report:")
    #print(report)
    #print("\nFeature Importance:")
    #for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
       # print(f"{feature}: {importance:.4f}")
    
    
    probabilities = predictor.predict_win_probability(match_info)
    response = {
        "team1": match_info['team1'],
        "team2": match_info['team2'],
        "team1_win_probability": probabilities['team1_win_probability']*100,
        "team2_win_probability": probabilities['team2_win_probability']*100,
        }

    return jsonify(response)

# Helper function to read CSV data
def read_csv_data(file_path):
    team1 = set()
    team2 = set()
    cities = set()

    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)

            for row in csv_reader:
                team1_name = row.get('team1')
                team2_name = row.get('team2')
                city_name = row.get('city')

                if team1_name:
                    team1.add(team1_name)
                if team2_name:
                    team2.add(team2_name)
                if city_name:
                    cities.add(city_name)

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], [], []

    return list(team1), list(team2), list(cities)


# Route to fetch dropdown data
@app.route('/dropdown_data', methods=['GET'])
def dropdown_data():
    try:
        csv_file_path = r'/Users/samuel/codemf/NewSMcric/FEHomePage/output2.csv'

        # Read team1, team2, and cities from the CSV file
        team1, team2, cities = read_csv_data(csv_file_path)

        # Prepare the data in the expected format
        data = {
            "team1": [{"name": team, "icon": "🏏"} for team in team1],
            "team2": [{"name": team, "icon": "🏏"} for team in team2],
            "cities": cities
        }

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": "Failed to fetch dropdown data", "message": str(e)}), 500


# Route to predict match outcome based on user input
@app.route('/predict', methods=['GET'])
def predict_form():
    # Serve the HTML form for user input
    return render_template('predict.html')


@app.route('/predict/result', methods=['POST'])
def predict_result():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate that data is not None
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Extract data with validation
        required_fields = [
            'team1', 'team2', 'city', 'required_runs', 'remaining_overs',
            'remaining_wickets', 'toss_winner', 'toss_decision', 'target_runs'
        ]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({"error": "Missing required fields", "fields": missing_fields}), 400

        # Convert numeric fields
        required_runs = int(data.get('required_runs', 0))
        remaining_overs = float(data.get('remaining_overs', 0))
        remaining_wickets = int(data.get('remaining_wickets', 0))
        target_runs = int(data.get('target_runs', 0))

        match_info = {
            'team1': data['team1'],
            'team2': data['team2'],
            'city': data['city'],
            'target_runs': target_runs,
            'target_overs': 20,
            'required_runs': required_runs,
            'remaining_overs': remaining_overs,
            'required_wickets': remaining_wickets,
            'toss_winner': data['toss_winner'],
            'toss_decision': data['toss_decision']
        }

        # Get predictions from the model
        predictionf = main(match_info)
        return predictionf

    except Exception as e:
        return jsonify({"error": "An error occurred while processing the request", "message": str(e)}), 500


# Route for live matches (optional, not used directly in the frontend)
@app.route('/live_matches')
def live_matches():
    url = "https://cricbuzz-cricket.p.rapidapi.com/matches/v1/recent"
    try:
        response = requests.get(url, headers={
            'x-rapidapi-key': "b4010528d4msh2f954e0d2c08bbcp1f65aejsn4044e0015e66",
            'x-rapidapi-host': "cricbuzz-cricket.p.rapidapi.com"
        })
        if response.status_code == 200:
            data = response.json()
            return jsonify(data)
        else:
            return jsonify({"error": "Failed to fetch live matches", "status_code": response.status_code}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "An error occurred while fetching live matches", "message": str(e)}), 500


# Route to display the homepage
@app.route('/')
def index():
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)