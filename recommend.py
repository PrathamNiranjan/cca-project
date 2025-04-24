# app/ml_models/crop_recommendation.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CropRecommender:
    def __init__(self, model_path='models/crop_recommender.pkl'):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path

    def prepare_training_data(self, soil_data, crop_details, crop_prices):
        """
        Prepare the data for training by combining soil data and crop suitability
        Returns:
            DataFrame with features and target variable (crop_id)
        """
        training_data = []

        for _, district_soil in soil_data.iterrows():
            district_id = district_soil['district_id']

            for _, crop in crop_details.iterrows():
                crop_id = crop['crop_id']

                is_suitable = (
                    district_soil['nitrogen'] >= crop['min_nitrogen'] and
                    district_soil['phosphorus'] >= crop['min_phosphorus'] and
                    district_soil['potassium'] >= crop['min_potassium'] and
                    district_soil['ph'] >= crop['min_ph'] and
                    district_soil['ph'] <= crop['max_ph'] and
                    district_soil['soil_type'] in crop['suitable_soil_types']
                )

                crop_district_prices = crop_prices[
                    (crop_prices['crop_id'] == crop_id) &
                    (crop_prices['district_id'] == district_id)
                ]
                avg_price = crop_district_prices['price_per_quintal'].mean() if not crop_district_prices.empty else 0

                if is_suitable:
                    training_data.append({
                        'district_id': district_id,
                        'soil_type': district_soil['soil_type'],
                        'nitrogen': district_soil['nitrogen'],
                        'phosphorus': district_soil['phosphorus'],
                        'potassium': district_soil['potassium'],
                        'ph': district_soil['ph'],
                        'rainfall': district_soil['rainfall'],
                        'land_size': np.random.uniform(1, 20),
                        'crop_id': crop_id,
                        'price': avg_price
                    })

        return pd.DataFrame(training_data)

    def train(self, soil_data, crop_details, crop_prices):
        """
        Train the crop recommendation model
        """
        training_data = self.prepare_training_data(soil_data, crop_details, crop_prices)

        # Consider removing district_id if it's not one-hot encoded
        X = training_data[['nitrogen', 'phosphorus', 'potassium', 'ph', 'rainfall', 'land_size']]
        y = training_data['crop_id']

        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)
        print(f"Model accuracy: {score:.4f}")

        self.save_model()

        return score

    def recommend_crops(self, district_id, soil_type, nitrogen, phosphorus, potassium, ph, rainfall, land_size,
                        crop_details, crop_prices, top_n=5):
        """
        Recommend crops based on soil conditions and land size
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")

        input_data = pd.DataFrame({
            'nitrogen': [nitrogen],
            'phosphorus': [phosphorus],
            'potassium': [potassium],
            'ph': [ph],
            'rainfall': [rainfall],
            'land_size': [land_size]
        })

        X_input_scaled = self.scaler.transform(input_data)

        crop_probs = self.model.predict_proba(X_input_scaled)[0]

        results = pd.DataFrame({
            'crop_id': self.model.classes_,
            'suitability_score': crop_probs
        }).sort_values(by='suitability_score', ascending=False)

        top_crops = results.head(top_n)
        recommendations = []

        for _, row in top_crops.iterrows():
            crop_id = int(row['crop_id'])
            crop_info = crop_details[crop_details['crop_id'] == crop_id].iloc[0]

            crop_district_prices = crop_prices[
                (crop_prices['crop_id'] == crop_id) &
                (crop_prices['district_id'] == district_id)
            ]
            latest_price = crop_district_prices['price_per_quintal'].iloc[-1] if not crop_district_prices.empty else 0

            recommendations.append({
                'crop_id': crop_id,
                'crop_name': crop_info['crop_name'],
                'suitability_score': row['suitability_score'],
                'season': crop_info['season'],
                'water_requirement': crop_info['water_requirement'],
                'growth_days': crop_info['growth_days'],
                'estimated_price': latest_price,
                'estimated_revenue': latest_price * land_size
            })

        return pd.DataFrame(recommendations)

    def save_model(self):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        saved_data = joblib.load(self.model_path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        print(f"Model loaded from {self.model_path}")


def create_and_train_model(soil_data_path, crop_details_path, crop_prices_path):
    soil_data = pd.read_csv(soil_data_path)
    crop_details = pd.read_csv(crop_details_path)
    crop_prices = pd.read_csv(crop_prices_path)

    recommender = CropRecommender()
    score = recommender.train(soil_data, crop_details, crop_prices)

    return recommender, score


if __name__ == "__main__":
    soil_data_path = 'data/soil_data.csv'
    crop_details_path = 'data/crop_details.csv'
    crop_prices_path = 'data/crop_prices_history.csv'

    recommender, score = create_and_train_model(soil_data_path, crop_details_path, crop_prices_path)

    crop_details = pd.read_csv(crop_details_path)
    crop_prices = pd.read_csv(crop_prices_path)

    recommendations = recommender.recommend_crops(
        district_id=5,
        soil_type='Red Loamy Soil',
        nitrogen=75,
        phosphorus=42,
        potassium=38,
        ph=6.6,
        rainfall=950,
        land_size=5,
        crop_details=crop_details,
        crop_prices=crop_prices,
        top_n=5
    )

    print("Recommended crops:")
    print(recommendations)
