import joblib
import json
import pandas as pd



def load_model_artifacts(model_dir):
    """Load all model artifacts"""

    # Load model
    model = joblib.load(f"{model_dir}/model.pkl")

    #Load encoders
    encoders = joblib.load(f"{model_dir}/encoders.pkl")

    # Load metadata
    with open(f"{model_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata['model_name']}(v{metadata['timestamp']})")

    return model, encoders, metadata


class TitanicPredictor:
    """Production-ready prediction pipeline"""

    def __init__(self, model_dir):
        self.model, self.encoders, self.metadata = load_model_artifacts(model_dir)
        self.feature_names = self.metadata['feature_names']

    def preprocess_single_passenger(self, passenger_data):
        """Preprocess a single passenger's data"""

        # Convert to DataFrame
        df = pd.DataFrame([passenger_data])

        # Handle missing value (simplified for demo)
        df['age'] = df['age'].fillna(df['age'].median())
        df['fare'] = df['fare'].fillna(df['fare'].median())
        df['embarked'] = df['embarked'].fillna('S')

        # Create engineered features
        df['family_size'] = df['sibsp'] + df['parch'] + 1
        df['is_alone'] = (df['family_size'] == 1).astype(int)

        # Extract title
        df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df['title'] = df['title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['title'] = df['title'].replace('Mlle', "Miss")
        df['title'] = df['title'].replace('Ms', "Miss")
        df['title'] = df['title'].replace('Mme', "Mrs")

        # Select and encode features
        X = df[['pclass', 'age', 'sex', 'fare', 'embarked', 'family_size', 'is_alone', 'title']].copy()

        # Apply label encoding
        for column in ['sex', 'embarked', 'title']:
            if column in self.encoders:
                try:
                    X[column] = self.encoders[column].transform(X[column])
                except ValueError:
                    # Handle unknown categories
                    X[column] = 0

        return X
    
    def predict(self, passenger_data):
        """Make prediction for a single passeger"""

        try:
            X = self.preprocess_single_passenger(passenger_data)
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0]

            return {
                'survival_prediction': int(prediction),
                'survival_probability': float(probability[1]),
                'confidence': float(max(probability))
            }
        except Exception as e:
            return {'error': str(e)}
        

model_dir = "models/titanic_survival_v20250817_130801"

# Test the predictor
predictor = TitanicPredictor(model_dir)

# Example passenger
test_passenger = {
    'pclass': 3, 
    'name': 'Doe, Mr. John',
    'sex': 'male',
    'age': 30,
    'sibsp': 0,
    'parch': 0,
    'fare': 8.05,
    'embarked': 'S'
}

result = predictor.predict(test_passenger)

print("Prediction result:", result)
