import json
import pandas as pd
import re

def debug_movie_mapping():
    """Debug movie ID mapping"""
    print("🔍 Debugging Movie ID Mapping")
    print("=" * 50)
    
    # Load ID mappings
    with open('../user2id.json', 'r') as f:
        user2id = json.load(f)
    with open('../item2id.json', 'r') as f:
        item2id = json.load(f)
    
    print(f"📊 Total users in user2id: {len(user2id)}")
    print(f"📊 Total items in item2id: {len(item2id)}")
    
    # Load movie information
    try:
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
        
        print(f"📊 Total movies in dataset: {len(movies_df)}")
        
        # Create a mapping from movie_id to movie info
        movie_info = {}
        for _, row in movies_df.iterrows():
            movie_info[int(row['movie_id'])] = {
                'title': row['title_clean'],
                'year': row['year'],
                'genres': row['genres'],
                'full_title': row['title']
            }
        
        print(f"📊 Total movies in movie_info: {len(movie_info)}")
        
        # Test some mappings
        print("\n🧪 Testing Movie ID Mappings:")
        print("-" * 30)
        
        # Test internal IDs that should map to real movies
        test_internal_ids = [0, 1, 2, 3, 4, 73, 84, 86, 52]
        
        for internal_id in test_internal_ids:
            # Find original movie ID
            original_movie_id = None
            for orig_id, internal_id_mapped in item2id.items():
                if internal_id_mapped == internal_id:
                    original_movie_id = orig_id
                    break
            
            if original_movie_id is not None:
                if original_movie_id in movie_info:
                    movie_data = movie_info[original_movie_id]
                    print(f"✅ Internal ID {internal_id} -> Original ID {original_movie_id} -> '{movie_data['title']}' ({movie_data['year']})")
                else:
                    print(f"❌ Internal ID {internal_id} -> Original ID {original_movie_id} -> NOT FOUND in movie_info")
            else:
                print(f"❌ Internal ID {internal_id} -> NO MAPPING FOUND")
        
        # Check what's in item2id
        print(f"\n📋 First 10 items in item2id:")
        print("-" * 30)
        for i, (orig_id, internal_id) in enumerate(list(item2id.items())[:10]):
            print(f"Original ID: {orig_id} -> Internal ID: {internal_id}")
        
        # Check what's in movie_info
        print(f"\n📋 First 10 movies in movie_info:")
        print("-" * 30)
        for i, (movie_id, movie_data) in enumerate(list(movie_info.items())[:10]):
            print(f"Movie ID: {movie_id} -> '{movie_data['title']}' ({movie_data['year']})")
        
        # Check for missing movies
        missing_movies = []
        for orig_id in item2id.keys():
            if int(orig_id) not in movie_info:
                missing_movies.append(orig_id)
        
        if missing_movies:
            print(f"\n⚠️  Missing movies in movie_info: {len(missing_movies)}")
            print(f"First 10 missing: {missing_movies[:10]}")
        else:
            print(f"\n✅ All movies in item2id are found in movie_info")
        
    except Exception as e:
        print(f"❌ Error loading movie information: {e}")

if __name__ == "__main__":
    debug_movie_mapping() 