import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_swiggy_data():
    """
    Converts MovieLens dataset into Swiggy-like food delivery context
    WHY: 
    - Demonstrates "extracting relevant information from historical data"
    - Shows ability to reframe external data for business problems
    """
    # Load public MovieLens dataset
    ratings = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
                          sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Convert movies → dishes, genres → cuisine types
    dishes = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.item', 
                         sep='|', encoding='latin-1', header=None)
    dishes = dishes[[0]][[1]].rename(columns={0: 'item_id', 1: 'dish_name'})
    
    # Add Swiggy-specific cuisine mapping
    cuisine_map = {
        'Comedy': 'Indian Street Food', 
        'Action': 'Quick Bites',
        'Drama': 'Fine Dining',
        'Horror': 'Late Night Snacks',
        'Sci-Fi': 'Healthy Bowls'
    }
    
    # Merge and clean
    data = pd.merge(ratings, dishes, on='item_id')
    data['cuisine'] = data['dish_name'].str.extract(r'\((.*?)\)')[0].map(cuisine_map).fillna('Other')
    
    # Business-relevant filtering
    popular_dishes = data['item_id'].value_counts()[data['item_id'].value_counts() > 50].index
    filtered_data = data[data['item_id'].isin(popular_dishes)]
    
    # Train-test split
    train, test = train_test_split(filtered_data, test_size=0.2, random_state=42)
    
    print(f"✅ Prepared {len(train)} Swiggy-style food interactions (users x dishes)")
    return train, test, cuisine_map

if __name__ == "__main__":
    train, test, cuisine_map = prepare_swiggy_data()
    # Save for model training
    train.to_csv('swiggy_train.csv', index=False)
