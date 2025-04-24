# app/ml_models/price_prediction.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import datetime
import os

class CropPricePredictor:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path if model_path else 'models/price_prediction_model.pkl'
        
        if os.path.exists(self.model_path):
            self.load_model()
        
    def prepare_data(self, data):
        """
        Prepare the data for training or prediction
        """
        # Extract year and month as features
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        
        # Create features and target
        X = data[['crop_id', 'district_id', 'year', 'month']]
        y = data['price_per_quintal']
        
        return X, y
    
    def train(self, prices_data):
        """
        Train the price prediction model
        
        Args:
            prices_data: DataFrame with crop_id, district_id, date, price_per_quintal
        """
        # Ensure date is datetime
        prices_data['date'] = pd.to_datetime(prices_data['date'])
        
        # Prepare data
        X, y = self.prepare_data(prices_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        score = self.model.score(X_test, y_test)
        print(f"Model R² score: {score:.4f}")
        
        # Save model
        self.save_model()
        
        return score
    
    def predict(self, crop_id, district_id, future_months=3):
        """
        Predict crop prices for the next few months
        
        Args:
            crop_id: ID of the crop to predict
            district_id: ID of the district to predict for
            future_months: Number of months to predict into the future
            
        Returns:
            DataFrame with date and predicted price
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Create dates for prediction
        today = datetime.date.today()
        future_dates = [today.replace(day=1) + datetime.timedelta(days=30*i) for i in range(future_months)]
        
        # Create prediction data
        prediction_data = []
        for date in future_dates:
            prediction_data.append({
                'crop_id': crop_id,
                'district_id': district_id,
                'year': date.year,
                'month': date.month,
                'date': date
            })
        
        # Convert to DataFrame
        prediction_df = pd.DataFrame(prediction_data)
        
        # Scale features
        X_pred = prediction_df[['crop_id', 'district_id', 'year', 'month']]
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        
        # Add predictions to DataFrame
        prediction_df['predicted_price'] = predictions
        
        return prediction_df[['date', 'predicted_price']]
    
    def save_model(self):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model and scaler
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Load model and scaler
        saved_data = joblib.load(self.model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        print(f"Model loaded from {self.model_path}")


def create_and_train_model(prices_data_path):
    """Helper function to create and train the model"""
    # Load data
    prices_data = pd.read_csv(prices_data_path)
    prices_data['date'] = pd.to_datetime(prices_data['date'])
    
    # Create and train model
    predictor = CropPricePredictor()
    score = predictor.train(prices_data)
    
    return predictor, score


if __name__ == "__main__":
    # For testing the module directly
    data_path = 'data/crop_prices_history.csv'
    predictor, score = create_and_train_model(data_path)
    
    # Test prediction
    predictions = predictor.predict(crop_id=1, district_id=5, future_months=6)
    print("Predicted prices:")
    print(predictions)