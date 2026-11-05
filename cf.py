import pandas as pd
import math
import os
import pickle
import zipfile
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()


def create_pseudo_ratings(movie_index,content_scores,movie_voters,user_ratings,user_means): #when movie doesn't exist in ratings.pkl, use reviews of other films

    # {key: movie_id, value: rating}

    pseudo_ratings={}

    # Get movies according to content scores, that appear on ratings.pkl


    valid_movies=[(movie_id,score) for movie_id,score in content_scores.items()
                  if movie_id!=movie_index and len(movie_voters.get(movie_id,{}))>0] #don't compare movie with itself, and check if it's included in movie_voters (ratings.pkl file)


    # sort according to content scores, get top-20

    sorted_valid_movies=sorted(valid_movies,key=lambda x:x[1],reverse=True)[:20] #sort according to score


    for movie_id,score in sorted_valid_movies: #for each movie

        current_movie_voters=movie_voters[movie_id] #get the voters of current movie

        for user_id,rating in current_movie_voters.items(): #for user,rating included in movie voter of current movie

            centered_rating=rating-user_means[user_id] #center rating (different users may be stricter on reviews)

            for cf_movie_id,cf_rating in user_ratings[user_id].items(): #other movies rated by the user

                if cf_movie_id == movie_index: #don't compare a movie with itself
                    continue

                cf_centered_rating=cf_rating-user_means[user_id] #center rating (different users may be stricter on reviews)

                if cf_movie_id not in pseudo_ratings: #if movie doesn't exist in dictionary

                     pseudo_ratings[cf_movie_id]=0.0 #initialize

                pseudo_ratings[cf_movie_id] +=centered_rating*cf_centered_rating*(score**2) #add scores, (score^2) for more content heavy

    pseudo_ratings={movie_id:score for movie_id,score in pseudo_ratings.items() #keep only positive ratings
                    if score>0}

    max_score=max(pseudo_ratings.values()) #get max rating

    for movie_id in pseudo_ratings:
        pseudo_ratings[movie_id] /= max_score #normalize ( so range is [0,1] )

    return pseudo_ratings


def find_item_based_similarity(movie_index,movie_voters,user_ratings,user_means,common_user_required): #movie based similarity (Collaborative Filtering), based on Pearson Correlation (mean-centering Cosine Similarity ( A * B / ||A|| * ||B|| ) )

    if movie_index not in movie_voters or len(movie_voters[movie_index])<5:  # if index doesn't exist in the dictionary, or movie has less than 5 voters

        return {}

    target_votes=movie_voters[movie_index] #voters for current movie ( {user_id:rating} )

    # Find average rating for each user

    # Dictionary {key: user_id, value: average rating}

    target_votes_centered={user_id: rating - user_means[user_id]
                           for user_id, rating in target_votes.items()
                           if user_id in user_ratings}

    target_norm=math.sqrt(sum(r ** 2 for r in target_votes_centered.values())) # ||A||

    if target_norm ==0:

        return {} #avoid division by 0

    movie_stats={} #{key: movie_id, value: [dot_product,norm,common_count] }

    #dot_products =  A * B

    #other_norms = ||B||

    #common_count: number of common reviews 2 movies share

    for user_id,rating in target_votes_centered.items(): # For each user with a rating in current movie

        if user_id in user_ratings: #If user exists in dictionary

            for user_movie_index,user_rating in user_ratings[user_id].items(): #Check user ratings on all films

                if user_movie_index!=movie_index: # don't compare the given movie

                    diff=user_rating-user_means[user_id] #mean-center user's rating

                    if user_movie_index not in movie_stats: #If movie doesn't exist in movie_stats

                        movie_stats[user_movie_index]=[diff*rating,diff**2,1]

                    else: #already exists

                        movie_stats[user_movie_index][0] += diff*rating
                        movie_stats[user_movie_index][1] += diff**2
                        movie_stats[user_movie_index][2] += 1



    similarities={}

    for movie_id,stats in  movie_stats.items():

        dot_product=stats[0]
        norm=stats[1]
        common_count=stats[2]

        norm=math.sqrt(norm)

        if common_count >= common_user_required: #at least (common_user_required) common reviews between the 2 movies

             if norm>0 and target_norm>0: #avoid division by zero

                 similarity=dot_product/ (norm *target_norm)

             else:
                     similarity=0.0

             if similarity >0: #only positive scores

                    similarities[movie_id]=similarity

    return similarities





def create_ratings_file(df): #function that creates file with ratings from users, on movies included in the tmdb_dataset.json file

    if os.path.exists("ratings.pkl"): #if file already exists, return
        return

    zip_path="ml-32m.zip"
    extract_dir="ml-32m"
    url="https://files.grouplens.org/datasets/movielens/ml-32m.zip"

    if not os.path.exists(zip_path) and not os.path.exists(extract_dir): #download zip file, if it doesn't exist

       try:
           response = session.get(url, timeout=10, verify=False)

           print("Downloading movieLens ratings...")

           with open(zip_path, "wb") as f: #"wb"= write binary
               for chunk in response.iter_content(chunk_size=50*1024*1024): #50 MB chunks

                    if chunk:
                         f.write(chunk)

           print(f"Downloaded {zip_path}")

       except requests.exceptions.RequestException as e:

           print(f"Network error during download: {e}")
           return

    if not os.path.exists(extract_dir): #extract useful files from zip (links and ratings)

        with zipfile.ZipFile(zip_path, "r") as zip_ref: #read zipfile

            zip_ref.extract(f"{extract_dir}/links.csv") #extract links
            zip_ref.extract(f"{extract_dir}/ratings.csv") #extract ratings

        print(f"Extracted links.csv and ratings.csv from: {zip_path}")


    links_path=os.path.join(extract_dir, "links.csv") #get file path
    ratings_path=os.path.join(extract_dir, "ratings.csv") #get file path

    #Match id's from TMDB dataset with MovieLens dataset id's. Only id's appearing in TMDB dataset are useful

    tmdb_ids=set(df["id"].dropna().astype(int)) #get all id's from TMDB, avoiding possible duplicates

    #Create a dictionary: {key: TMDB_id, value: movie_index}, for TMDB dataset

    tmdb_ids_to_index= {movie_id: index for index,movie_id in enumerate(df["id"].dropna().astype(int))} #get all id's from TMDB, avoiding possible duplicates

    links=pd.read_csv(links_path) #read links

    #Omit links not appearing in TMDB dataset

    useful_links=links[links["tmdbId"].isin(tmdb_ids)]

    #Create a dictionary: {key: MovieLens_id, value: movie_index} for MovieLens dataset

    movieLens_ids_to_index={int (row["movieId"]): tmdb_ids_to_index[int(row["tmdbId"])]
                            for _,row in useful_links.iterrows()}

    useful_movieLens_ids=set(movieLens_ids_to_index.keys())  #get all id's

    # Create a dictionary: {key: movie_index, value: {user_id:rating}, for item-based collaborative filtering

    movie_voters={}

    # Create a dictionary: {key: user_id, value: {movie_index:rating}, for user-based collaborative filtering

    user_ratings={}

    #We have 32 million ratings, process 1 million each time

    chunk_size=1_000_000
    print("Processing ratings . . .")

    for chunk in pd.read_csv(ratings_path, usecols=["userId","movieId","rating"],chunksize=chunk_size):

        useful_chunk=chunk[chunk["movieId"].isin(useful_movieLens_ids)]  #Only if id is in useful_movieLens_ids

        for row in useful_chunk.itertuples(index=False):

            movie_id=int(row.movieId)
            movie_index=movieLens_ids_to_index[movie_id]
            user_id=int(row.userId)
            rating=float(row.rating)

            if movie_index not in movie_voters:

                movie_voters[movie_index]={} #create empty slot, to store dictionary value

            movie_voters[movie_index][user_id]=rating #store rating for current movie index and user id

            if user_id not in user_ratings:

                user_ratings[user_id]={} #create empty slot, to store dictionary value

            user_ratings[user_id][movie_index]=rating #store rating for current user id and movie index

    # Dictionary {key: user_id, value: average rating}

    user_means = {user_id: sum(user_ratings[user_id].values()) / len(user_ratings[user_id])
                      for user_id in user_ratings}

    with open("ratings.pkl", "wb") as f:
        pickle.dump((movie_voters,user_ratings,user_means),f,protocol=pickle.HIGHEST_PROTOCOL) #write data to pickle file (serialized)


    print("Deleting temporary files (zip, csv)")

    if os.path.exists(zip_path): #delete zip file

        os.remove(zip_path)

    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir): #for each extracted file
            os.remove(os.path.join(extract_dir, file)) #remove file
        os.rmdir(extract_dir) #remove directory

    print("Ratings.pkl successfully created")










