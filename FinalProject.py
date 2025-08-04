import numpy as np
import pandas as pd
import requests
import time
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import pickle
import os

# Set global font settings
API_KEY = '6c2be6da3ffa35826f19e9d9afd7d561'
CACHE_FILE = 'tmdb_api_cache.pkl'

class TMDBAPICache:
    """Cache system for TMDB API responses to minimize requests"""
    def __init__(self, cache_file=CACHE_FILE):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
    def _load_cache(self):
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {'search': {}, 'credits': {}}
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def get_cached_search(self, title, year):
        key = f"{title}_{year}"
        return self.cache['search'].get(key)
    
    def set_cached_search(self, title, year, result):
        key = f"{title}_{year}"
        self.cache['search'][key] = result
    
    def get_cached_credits(self, tmdb_id):
        return self.cache['credits'].get(str(tmdb_id))
    
    def set_cached_credits(self, tmdb_id, result):
        self.cache['credits'][str(tmdb_id)] = result

api_cache = TMDBAPICache()

def get_tmdb_id(title, year, max_retries=3):
    """Fetch TMDB ID with retry mechanism and caching"""
    cached = api_cache.get_cached_search(title, year)
    if cached is not None:
        return cached
    
    search_url = 'https://api.themoviedb.org/3/search/movie'
    params = {
        'api_key': API_KEY,
        'query': title,
        'year': year,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                results = response.json().get('results', [])
                tmdb_id = results[0]['id'] if results else None
                api_cache.set_cached_search(title, year, tmdb_id)
                return tmdb_id
            elif response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 10))
                time.sleep(wait_time)
                continue
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {title}: {str(e)}")
            time.sleep(2 ** attempt)
    
    api_cache.set_cached_search(title, year, None)
    return None

def get_cast_info(tmdb_id, max_retries=3):
    """Fetch cast information with caching"""
    if pd.isna(tmdb_id):
        return 0
    
    cached = api_cache.get_cached_credits(tmdb_id)
    if cached is not None:
        return cached
    
    credits_url = f'https://api.themoviedb.org/3/movie/{tmdb_id}/credits'
    params = {'api_key': API_KEY}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(credits_url, params=params, timeout=10)
            if response.status_code == 200:
                cast_count = len(response.json().get('cast', []))
                api_cache.set_cached_credits(tmdb_id, cast_count)
                return cast_count
            elif response.status_code == 429:
                wait_time = int(response.headers.get('Retry-After', 10))
                time.sleep(wait_time)
                continue
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for tmdb_id={tmdb_id}: {str(e)}")
            time.sleep(2 ** attempt)
    
    api_cache.set_cached_credits(tmdb_id, 0)
    return 0

def analyze_movies_data(input_csv, output_csv='movies_analysis_results.csv'):
    """Main analysis pipeline with data caching"""
    print("Loading movie data...")
    movies = pd.read_csv(input_csv)
    movies['release_date'] = pd.to_datetime(movies['release_date'])
    movies = movies[
        (movies['release_date'].dt.year.between(2005, 2025)) &
        (movies['revenue'] > 0) &
        (movies['budget'] > 0)
    ].copy()
    movies['year'] = movies['release_date'].dt.year
    
    if Path(output_csv).exists():
        print("Existing analysis results detected, loading...")
        return pd.read_csv(output_csv)
    
    if 'tmdb_id' not in movies.columns:
        movies['tmdb_id'] = None
    
    print("\nFetching TMDB IDs (first run may take time, results will cache)...")
    for idx, row in tqdm(movies.iterrows(), total=len(movies), desc="Querying Movie IDs"):
        if pd.isna(row['tmdb_id']):
            movies.at[idx, 'tmdb_id'] = get_tmdb_id(row['title'], row['year'])
            time.sleep(0.25)
    
    if 'cast_count' not in movies.columns:
        movies['cast_count'] = 0
    
    print("\nFetching cast information...")
    for idx, row in tqdm(movies.iterrows(), total=len(movies), desc="Querying Cast Size"):
        if pd.isna(row['cast_count']) or row['cast_count'] == 0:
            movies.at[idx, 'cast_count'] = get_cast_info(row['tmdb_id'])
            time.sleep(0.25)
    
    # Feature engineering
    movies['star_power'] = movies['vote_average'] * np.log1p(movies['vote_count'])
    movies['is_action_adv'] = movies['genres'].apply(lambda x: int(('Action' in x) and ('Adventure' in x)))
    
    movies.to_csv(output_csv, index=False)
    api_cache.save_cache()
    print(f"\nAnalysis results saved to {output_csv}")
    return movies

def run_analysis(movies):
    """Generate analytical insights and visualizations"""
    action_adv = movies[movies['is_action_adv'] == 1]
    others = movies[movies['is_action_adv'] == 0]
    
    print("\nMovie Industry Analysis (2005-2025):")
    print("="*50)
    print(f"Total Movies Analyzed: {len(movies):,}")
    print(f"Action-Adventure Films: {len(action_adv):,} ({len(action_adv)/len(movies):.1%})")
    print(f"Other Genres: {len(others):,}")
    print("\nRevenue Comparison (Mean Values):")
    print(f"- Action-Adventure: ${action_adv['revenue'].mean()/1e6:.1f} million")
    print(f"- Other Genres: ${others['revenue'].mean()/1e6:.1f} million")
    print(f"Revenue Multiplier: {action_adv['revenue'].mean()/others['revenue'].mean():.1f}x")
    
    # Regression Analysis
    X = movies[['budget', 'popularity', 'is_action_adv', 'star_power', 'cast_count']].copy()
    X['budget'] = np.log1p(X['budget'])
    y = np.log1p(movies['revenue'])
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Action-Adventure', 'Other Genres'], 
           [action_adv['revenue'].mean()/1e6, others['revenue'].mean()/1e6],
           color=['#1f77b4', '#ff7f0e'])
    plt.ylabel('Average Revenue (Millions USD)')
    plt.title('Genre Revenue Comparison 2005-2025')
    
    plt.subplot(1, 2, 2)
    features = ['Budget (log)', 'Popularity', 'Action-Adventure', 'Star Power', 'Cast Size']
    plt.barh(features, model.coef_)
    plt.title('Revenue Drivers Analysis')
    plt.xlabel('Standardized Coefficient Impact')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('movie_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.figure(figsize=(8, 8))
    abs_coefs = np.abs(model.coef_)
    weights = abs_coefs / np.sum(abs_coefs)
    plt.pie(weights, labels=features, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Relative Influence of Variables')
    plt.tight_layout()
    plt.savefig('relative_influence_pie_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    print("\nRegression Analysis Results:")
    print("="*50)
    print(f"Budget Coefficient: {model.coef_[0]:.4f} (1 SD increase → +{model.coef_[0]*100:.1f}% revenue)")
    print(f"Popularity Coefficient: {model.coef_[1]:.4f}")
    print(f"Action-Adventure Coefficient: {model.coef_[2]:.4f} (+{np.expm1(model.coef_[2])*100:.1f}% vs other genres)")
    print(f"Star Power Coefficient: {model.coef_[3]:.4f}")
    print(f"Cast Size Coefficient: {model.coef_[4]:.4f}")

if __name__ == "__main__":
    input_file = '/Users/aaron/Desktop/Data Science - Columbia Uni./Programms/safe_cleaned_dataset.csv'
    output_file = 'movies_analysis_results.csv'
    
    movies_data = analyze_movies_data(input_file, output_file)
    run_analysis(movies_data)
    
    print("\nAnalysis Complete! Results saved to:")
    print(f"- Data File: {output_file}")
    print(f"- Visualization: movie_analysis_results.png")
