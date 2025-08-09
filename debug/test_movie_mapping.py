import json
import pandas as pd
import re

# Load ID mappings
with open('../user2id.json', 'r') as f:
    user2id = json.load(f)
with open('../item2id.json', 'r') as f:
    item2id = json.load(f)

print(f"Total items in item2id: {len(item2id)}")

# Load movie information
movies_df = pd.read_csv('../ml-1m/movies.dat', sep='::', engine='python',
                        names=['movie_id', 'title', 'genres'], encoding='latin-1')

# Parse title and year
def extract_title_year(title):
    match = re.match(r'(.+?)\s*\((\d{4})\)', title)
    if match:
        return match.group(1).strip(), int(match.group(2))
    return title, None

movies_df[['title_clean', 'year']] = movies_df['title'].apply(
    lambda x: pd.Series(extract_title_year(x))
)

# Create a mapping from movie_id to movie info
movie_info = {}
for _, row in movies_df.iterrows():
    movie_info[int(row['movie_id'])] = {
        'title': row['title_clean'],
        'year': row['year'],
        'genres': row['genres'],
        'full_title': row['title']
    }

print(f"Total movies in movie_info: {len(movie_info)}")

# Test specific internal IDs
test_ids = [0, 1, 2, 73, 84, 86, 52]

for internal_id in test_ids:
    print(f"\nTesting internal ID: {internal_id}")
    
    # Find original movie ID
    original_movie_id = None
    for orig_id, internal_id_mapped in item2id.items():
        if internal_id_mapped == internal_id:
            original_movie_id = orig_id
            break
    
    if original_movie_id is not None:
        original_movie_id_int = int(original_movie_id)
        if original_movie_id_int in movie_info:
            movie_data = movie_info[original_movie_id_int]
            print(f"  ✅ Found: {movie_data['title']} ({movie_data['year']})")
        else:
            print(f"  ❌ Original ID {original_movie_id_int} not in movie_info")
    else:
        print(f"  ❌ No mapping found")

# Show first few mappings
print(f"\nFirst 10 mappings in item2id:")
for i, (orig_id, internal_id) in enumerate(list(item2id.items())[:10]):
    print(f"  {orig_id} -> {internal_id}")

# Show first few movies
print(f"\nFirst 10 movies in movie_info:")
for i, (movie_id, movie_data) in enumerate(list(movie_info.items())[:10]):
    print(f"  {movie_id} -> {movie_data['title']}") 