from concurrent.futures import ThreadPoolExecutor
import questionary
import requests
import time
import keys
import datetime



session = requests.Session()

from normalizations import *
from cf import *
from word2vec import create_movie_vectors_file, find_movie_vectors_similarity,create_doc_freq,text_to_words



def build_overview_index(df):


    # words that don't provide useful information for overview similarity

    corpus=[text_to_words(text,useStopWords=True,useStemming=True) for text in df["overview"]] #list of lists, each list is an overview

    #Calculate DF (Document Frequency)

    doc_freq=create_doc_freq(corpus) #dictionary that holds frequency of each word in the document

    #Caculate TF-IDF (Term Frequency - Inverse Document Frequency)

    N=len(corpus) #number of total movies

    inverted_index={} #dictionary that will hold for each word: 1) In which movies it appears 2) Its weight for each movie e.g. "batman": [(0,4.56),(15,3.12)] movies:0, 15 , weights:4.56, 3.12

    movie_norms=np.zeros(N) #array that stores denominator of Cosine Similarity for each movie
    movie_TF_IDF=[] #TF_IDF for each movie

    for movie_id,text in enumerate(corpus): #for each movie and text in the corpus

        tf={} #dictionary that holds how many times a word appears in an overview

        for w in text: #for each word in an overview

            if w not in tf: #if a word is not in the dictionary
                tf[w]=1 #Set frequency as 1
            else:
                tf[w]+=1 #if a word is in the dictionary, increment its frequency

        TF_IDF_dict={} #each movie has its own TF-IDF dictionary

        norm_sq=0.0 #denominator of Cosine Similarity ( A * B / ||A|| * ||B|| )

        for word,count in tf.items(): #for each word and count in TF

            #TF-IDF= TF * IDF
            #TF: how many times a word appears in a certain movie overview
            #IDF: takes into account how many times a word appears in the entire document
            #IDF = log( Number of different overviews (movies) / times a word appears in the entire document)

            #if doc_freq[word] > 1:  # if a word appears only one time in the document, remove it (no use)

                TF=count
                IDF=math.log(N/(doc_freq[word]))

                TF_IDF=TF*IDF
                TF_IDF_dict[word]=TF_IDF #store TF_IDF value for each word

                norm_sq+=TF_IDF*TF_IDF

        movie_TF_IDF.append(TF_IDF_dict) #store TF_IDF for each movie
        movie_norms[movie_id]=math.sqrt(norm_sq)

        #Set inverted index dictionary

        for word,weight in TF_IDF_dict.items(): #for each word and weight in dictionary

                if word not in inverted_index: #if word not in dictionary
                    inverted_index[word]=[] #create empty slot

                inverted_index[word].append((movie_id,weight)) #for each word, store movie id and its weight

    return inverted_index,movie_TF_IDF,movie_norms



def find_movie_overview_similarity(movie_index,inverted_index,movie_TF_IDF,movie_norms):


    #Instead of comparing a movie with all other movies, compare only with movies with at least 1 common word

    #Calculate Cosine Similarity using the inverted index dictionary

    dot_products={} #dictionary which will hold the scores for each movie

    #Calculate numerator of Cosine Similarity ( A * B / ||A|| * ||B|| )

    for word,weight in movie_TF_IDF[movie_index].items(): #for each word and its weight in movie i

        if word in inverted_index: #if the word exists in dictionary

            for movie, movie_weight in inverted_index[word]: #for each movie and its weight in dictionary

                 if movie_index!=movie: #don't compare with itself

                    if movie not in dot_products: #if a movie is not in the dictionary

                         dot_products[movie]=weight*movie_weight #multiply the weights

                    else:

                        dot_products[movie]=dot_products[movie]+(weight*movie_weight) #multiply the weights

    similarities={} #dictionary which will hold similarity for each movie

    norm=movie_norms[movie_index] #denomenator of Cosine Similarity of movie i

    for movie,dot_product in dot_products.items(): #for each movie and its score in dictionary

        movie_norm=movie_norms[movie] #get norm of movie

        if norm>0 and movie_norm>0: #avoid - * - = +

            similarity=dot_product/(norm*movie_norm) #Cosine Similarity
            similarities[movie]=similarity

    return similarities


def build_genre_index(df):

    movie_genres=[set(genres) for genres in df["genre_ids"]] #load genres onto a set, for each movie

    #Logic is: genre->movies e.g. Action ->[0, 1, 3] Comedy -> [0, 1]  Sci-Fi -> [0, 3] Drama ->[2]  (Inverted Index)

    genre_index={} #initialize emtpy dictionary, saving movies for each genre

    for movie_id,genres in enumerate(movie_genres): #for each movie

        for genre in genres: #for each genre of the current movie

            if genre not in genre_index: #if genre not in dictionary, create empty slot for movies to be added

                 genre_index[genre]=[]

            genre_index[genre].append(movie_id) #append movie id to current unique genre

    #Find IDF for each genre type (genres that are rare have more value than common genres)

    genre_weights = {}  # initialize emtpy dictionary, saving genre weights for each genre

    N = len(df)  # number of different movies

    for genre,movie_list in genre_index.items(): #for each genre and its movie list in dictionary

        genre_weights[genre]=math.log(N/len(movie_list)) #Calculate its IDF (Total movies / Movies with this genre)

    #calculate norms for each movie ( ||A|| ), denominator of Cosine Similarity

    movie_weights=np.zeros(N)
    for movie_id, genres in enumerate (movie_genres):
        movie_weight=sum(genre_weights[genre]**2 for genre in genres) #IDF_weight^2 for each genre in a movie
        movie_weights[movie_id]=math.sqrt(movie_weight)

    return movie_genres,genre_index,genre_weights,movie_weights


def find_movie_genre_similarity(movie_index,movie_genres,genre_index,genre_weights,movie_weights):

    # Find genre scores

    genre_scores={} #initialize emtpy dictionary, saving genres weight for each movie, numerator of Cosine Similarity


    for genre in movie_genres[movie_index]: #for each genre in a movie

        genre_weight=genre_weights[genre] #IDF weight for this genre

        for movie_id in genre_index[genre]: #for each movie in this genre

              if movie_id!=movie_index: #don't compare movie with itself

                   if movie_id not in genre_scores: #if a movie is not in the dictionary

                       genre_scores[movie_id]=genre_weight*genre_weight #This is the first common genre found, multiply the weights

                   else:

                       genre_scores[movie_id]+=genre_weight*genre_weight # if a movie is in the dictionary, multiply the weights

    #Calculate cosine similarity

    similarities={} #dictionary which will hold similarity for each movie

    movie_weight=movie_weights[movie_index] #weight of movie i

    for movie_j,genre_score in genre_scores.items():

          movie_j_weight=movie_weights[movie_j]
          similarity=genre_score/(movie_weight*movie_j_weight) #genre score / || genres of movie i || * || genres of movie movie_id ||
          similarities[movie_j]=similarity

    return similarities

def movies_share_collection(movie_index,df): #returns np.array of movies sharing the same collection

    collection_ids = df["collection_id"].to_numpy()

    movie_collection_id=df.loc[movie_index, "collection_id"] #locate collection_id based on its movie_index

    if not pd.isna(movie_collection_id):

        mask=np.where(collection_ids==movie_collection_id)[0] #keep only movies from the same collection

        return mask[mask!=movie_index]

    return np.array([])

def movies_share_director(movie_index,df):#returns np.array of movies sharing the same director

    director_ids = df["director"].to_numpy()

    movie_director_id=df.loc[movie_index, "director"]

    if not pd.isna(movie_director_id):

        mask=np.where(director_ids==movie_director_id)[0] #keep only movies from the same director

        return mask[mask!=movie_index]

    return np.array([])

def build_cast_index(df): #function returns dictionary: {key:actor_id, value: {key: movie_id, value: position} }

    #position: Protagonist (position 1),should have more weight than an actor in other positions

    cast_index={}

    for movie_id,cast in enumerate(df["cast"]): #for each movie and its cast

        if not isinstance(cast,list):

            continue

        for position,actor_id in enumerate(cast): #for each actor in cast

            if actor_id not in cast_index: #if actor is not in dictionary

                cast_index[actor_id]={} #create empty slot

            if movie_id not in cast_index[actor_id]: #if movie isn't stored for actor

                cast_index[actor_id][movie_id]=position


    return cast_index

def find_common_actors(movie_index,df,cast_index): #function returns number of common actors between movies {key: movie_id, value: number of common actors}

    movie_cast=df.loc[movie_index, "cast"] #get cast for current movie

    actor_scores={}

    if not isinstance(movie_cast,list):
        return {}

    for position,actor_id in enumerate(movie_cast): #for each actor

        weight = (10 - position) / 10  # we have 10 actors, if position=0, weight=1, if position=2, weight=0.8, if position=9, weight=0.1

        for movie_id in cast_index[actor_id]:  #for each movie current actor plays in

            if movie_id!=movie_index: #don't compare movie with itself

                 if movie_id not in actor_scores: #if movie doesn't exist in dictionary

                     actor_scores[movie_id]=weight #this is the first common actor found

                 else: #already exists in dictionary

                    actor_scores[movie_id]+=weight

    return actor_scores

def find_content_based_similarity(movie_index,overview_scores,genre_scores,word2vec_scores):

    all_candidates_movies = (set(overview_scores) | set(genre_scores) | set(word2vec_scores)) - {movie_index} #union of all scores (check only movies that have >0 for 1 of the two scores),don't compare movie with itself (-movie index)

    content_scores = {}

    for movie_id in all_candidates_movies:

        # check if movie shares a genre or something in its overview

        has_overview = movie_id in overview_scores
        has_genre = movie_id in genre_scores
        has_word2vec = movie_id in word2vec_scores

        overview_similarity = 0.0
        genre_similarity = 0.0
        word2vec_similarity = 0.0

        # content score

        if has_overview:

            overview_z = standardization(overview_scores[movie_id], overview_mean, overview_std)
            overview_similarity = sigmoid(overview_z)

        if has_genre:

            genre_z=standardization(genre_scores[movie_id], genre_mean, genre_std)
            genre_similarity = sigmoid(genre_z)

        if has_word2vec:
            word2vec_z=standardization(word2vec_scores[movie_id],word2vec_mean,word2vec_std)
            word2vec_similarity=sigmoid(word2vec_z)

        content_score = (0.35 * overview_similarity) + (0.25 * genre_similarity) + (0.4 * word2vec_similarity)

        if not has_genre:  # no common genre

            content_score = content_score * 0.75 # penalty (0.75)

        content_scores[movie_id]=content_score

    return content_scores

def find_overall_similarity(movie_index,cf_item_scores,content_scores,movie_voters,user_ratings,user_means,actor_scores):

    same_collection_movies = set(movies_share_collection(movie_index, df))

    same_director_movies = set(movies_share_director(movie_index, df))

    pseudo_cf_item=False

    if not cf_item_scores: #calculate pseudo-scores if they don't exist

        pseudo_cf_item=True

        cf_item_scores = create_pseudo_ratings(movie_index, content_scores, movie_voters, user_ratings,user_means)

    all_candidates_movies=(set(content_scores) | set(cf_item_scores)) - {movie_index} #union of all scores (check only movies that have >0 for 1 of the two scores),don't compare movie with itself (-movie index)

    similarities=[]

    for movie_id in all_candidates_movies:

        has_content=movie_id in content_scores
        has_cf_item=movie_id in cf_item_scores

        common_actor_weight = actor_scores.get(movie_id, 0) #get common actors for current movie

        if not pseudo_cf_item: #if real cf_item scores exist

            if  has_cf_item and has_content: #exists in ratings.pkl and content_scores

                 cf_item_z = standardization(cf_item_scores[movie_id], cf_item_mean, cf_item_std)

                 cf_item_similarity = sigmoid(cf_item_z)

                 content_score=content_scores[movie_id]

                 final_score=(0.4*content_score) + (0.6*cf_item_similarity)

            elif has_content: #doesn't exist in ratings.pkl, only content score available

                 content_score = content_scores[movie_id]

                 final_score=0.85*content_score

            else: #only exists in cf_item, no content score available


                 cf_item_z = standardization(cf_item_scores[movie_id], cf_item_mean, cf_item_std)

                 cf_item_similarity = sigmoid(cf_item_z)

                 final_score=0.7*cf_item_similarity

        if pseudo_cf_item: #if real cf_item scores don't exist

            if has_content and has_cf_item: #exists in ratings.pkl and content_scores

                content_score = content_scores[movie_id]
                pseudo_cf_item_similarity = cf_item_scores[movie_id]
                final_score = (0.75* content_score) + (0.25* pseudo_cf_item_similarity)

            elif has_content: #doesn't exist in ratings.pkl, only content score available

                content_score = content_scores[movie_id]
                final_score = 0.85 * content_score

            else: #only exists in cf_item, no content score available
                pseudo_cf_item_similarity = cf_item_scores[movie_id]
                final_score=0.3*pseudo_cf_item_similarity

        #Common actors bonus

        actors_bonus = 0.01 * common_actor_weight  #max is 5.5%
        final_score+=actors_bonus
        final_score = min(final_score, 1.0) # so final score doesn't surpass 100 %


        if movie_id in same_collection_movies:  # if movies belong in the same collection (franchise)

            final_score += 0.20
            final_score = min(final_score, 1.0)  # so final score doesn't surpass 100 %

        if movie_id in same_director_movies: # if movies has the same director

            final_score += 0.05
            final_score = min(final_score, 1.0) # so final score doesn't surpass 100 %


        similarities.append((final_score, movie_id))

    similarities.sort(reverse=True)
    return similarities

def calculate_statistics(df,overview_index,movieTF_IDF,movie_norms,movie_genres,genre_index,genre_weights,movie_weights,movie_voters,user_ratings,user_means,movie_vectors): #function that calculates means and std over a number of samples (used for standardization)


    if os.path.exists("statistics.pkl"):
        return

    print("Processing samples for statistics calculations (mean & std)  . . .")

    overview_samples = []
    genre_samples = []
    cf_item_samples = []
    word2vec_samples = []

    sample_size=2000 #2.000 movies

    sample_movies=np.random.choice(len(df),sample_size,replace=False)

    for movie_index in sample_movies:

        overview_scores = find_movie_overview_similarity(movie_index, overview_index, movieTF_IDF, movie_norms)

        genre_scores = find_movie_genre_similarity(movie_index, movie_genres, genre_index, genre_weights, movie_weights)

        cf_item_scores = find_item_based_similarity(movie_index, movie_voters, user_ratings, user_means,10)

        word2vec_scores=find_movie_vectors_similarity(movie_index,movie_vectors)

        overview_samples.extend(overview_scores.values())
        genre_samples.extend(genre_scores.values())
        cf_item_samples.extend(cf_item_scores.values())
        word2vec_samples.extend(word2vec_scores.values())

    #calculate mean

    overview_mean=np.mean(overview_samples)
    genre_mean=np.mean(genre_samples)
    cf_item_mean=np.mean(cf_item_samples)
    word2vec_mean=np.mean(word2vec_samples)

    #calculate std

    overview_std=np.std(overview_samples)
    genre_std=np.std(genre_samples)
    cf_item_std=np.std(cf_item_samples)
    word2vec_std=np.std(word2vec_samples)

    statistics=(overview_mean,genre_mean,cf_item_mean,word2vec_mean,overview_std,genre_std,cf_item_std,word2vec_std)

    with open("statistics.pkl", "wb") as f:
        pickle.dump(statistics,f,protocol=pickle.HIGHEST_PROTOCOL)

    print("statistics.pkl successfully created")



def fetch_movie_details(movie_id): #helper function that returns details for each movie (cast, director, if it belongs in a collection)
    api_key = keys.TMDB_API_KEY
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&append_to_response=credits"

    try:
        response = session.get(url, timeout=10)

        if response.status_code == 200:  # if success
            data = response.json()

            #Collection
            collection = data.get("belongs_to_collection")
            collection_id=collection["id"] if collection else None  # store collection id, else None

            #Director
            director=None
            crew=data.get("credits",{}).get("crew",[])

            for person in crew:

                if person.get("job") == "Director":
                    director=person.get("id")
                    break

            #Main actors
            cast=data.get("credits",{}).get("cast",[])
            actor_ids=[person.get("id") for person in cast[:10]] #10 first actors

            return collection_id, director, actor_ids

        elif response.status_code == 429:

            print("Rate limit reached, waiting 5 seconds...")
            time.sleep(5)
            return fetch_movie_details(movie_id)  # repeat
        else:

            return None,None,[]

    except requests.exceptions.RequestException as e:

        print(f"Network error for movie {movie_id}: {e}")
        return None,None,[]



def create_json(): #function that creates json file

    if os.path.exists("tmdb_dataset.json"): #if file already exists, return

        return

    api_key=keys.TMDB_API_KEY
    read_key=keys.TMDB_READ_TOKEN

    movie_list=[]

    current_year=datetime.datetime.now().year
    start_year=current_year-95

    for year in range(current_year, start_year, -1): #Reverse order so new movies appear on top when searching

        print(f"Collecting movies from year: {year}")

        page = 1
        while page <= 500:  # number of pages

            if year < 1985:  # more strict on older movies

                url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&primary_release_year={year}&sort_by=popularity.desc&vote_count.gte=300&page={page}"  # descending by popularity, at least 300 votes

            else:

                url = f"https://api.themoviedb.org/3/discover/movie?api_key={api_key}&primary_release_year={year}&sort_by=popularity.desc&vote_count.gte=100&page={page}"  # descending by popularity, at least 100 votes

            try:

                response = session.get(url, timeout=10)

                if response.status_code == 200:  # if success

                    data = response.json()
                    results = data.get("results", [])

                    if results:

                        movie_list.extend(results)  # use extend instead of append to avoid loop
                        page += 1  # move to next page

                    else:

                        break  # no more movies for current year

                elif response.status_code == 429:

                    print("Rate limit reached, waiting 5 seconds...")
                    time.sleep(5)

                else:

                    print(f"Error on year: {year}, page: {page}, Status code: {response.status_code}")
                    break

            except requests.exceptions.RequestException as e:

                print(f"Network Error on page: {page}, {e}")
                time.sleep(5)
                continue

    df = pd.DataFrame(movie_list) #convert to Pandas DataFrame

    df.drop_duplicates(subset=["id"], inplace=True)  # avoid possible duplicates based on same id

    print("Collecting movie details for each movie...")

    movie_ids=df["id"].tolist() #in order to field that says if movies are in a franchise

    with ThreadPoolExecutor() as executor: #use multiple threads at once, for faster results

        movie_details = list(executor.map(fetch_movie_details, movie_ids))

    df["collection_id"]=[details[0] for details in movie_details]
    df["director"]=[details[1] for details in movie_details]
    df["cast"]=[details[2] for details in movie_details]

    columns=['genre_ids','id','title','overview','vote_average','vote_count',"release_date","collection_id","director","cast"] #keep only useful columns

    df = df.reindex(columns=columns)  #avoid error if NaN field

    df["year"] = df["release_date"].astype(str).str.slice(0, 4)  # get only year of release date

    df["display_title"] = df.apply(  # make title appear release year too
        lambda row: f"{row['title']} ({row['year']})"
        if row["year"] and row["year"] != "nan"
        else row["title"],
        axis=1,
    )


    df.drop_duplicates(subset=["display_title"], inplace=True) #avoid possible duplicates based on same title

    df.reset_index(drop=True, inplace=True) #reset indices after duplicate drops

    #save as json to avoid API calls
    df.to_json("tmdb_dataset.json", index=False)

    print("tmdb_dataset.json successfully created")



def run_ui():
    print("\n" + "=" * 50)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 50 + "\n")

    style = questionary.Style([
        ("qmark", "fg:cyan bold"),
        ("question", "fg:white bold"),
        ("answer", "fg:black"),
        ("pointer", "fg:black"),
        ("highlighted", "fg:black bold"),
        ("selected", "fg:green"),
        ("text", "fg:white"),
    ])


    selected_title=questionary.autocomplete("Insert movie title (Use arrows): ",choices=available_titles,
        ignore_case=True,
        match_middle=True,
        style=style
        ).ask()

    if not selected_title:
        print("No movie title was selected")
        return
    try:

        movie_index= df[df["display_title"] == selected_title].index[0] #find movie index according to title

    except IndexError:

        print(f"\nMovie '{selected_title}' was not found.")
        print("Please try again.\n")
        return

    print(f"\n Searching recommendations for: '{selected_title}'...\n")

    overview_scores = find_movie_overview_similarity(movie_index,overview_index,movieTF_IDF,movie_norms)

    genre_scores = find_movie_genre_similarity(movie_index,movie_genres,genre_index,genre_weights,movie_weights)

    word2vec_scores = find_movie_vectors_similarity(movie_index, movie_vectors)

    content_scores=find_content_based_similarity(movie_index, overview_scores, genre_scores, word2vec_scores)

    cf_item_scores = find_item_based_similarity(movie_index,movie_voters,user_ratings,user_means,10)

    actor_scores=find_common_actors(movie_index,df,cast_index)

    recommendations = find_overall_similarity(movie_index,cf_item_scores,content_scores,movie_voters,user_ratings,user_means,actor_scores)

    print("=" * 50)
    print(f" TOP 20 RECOMMENDATIONS FOR: {selected_title.upper()}")
    print("=" * 50)

    for similarity, m_id in recommendations[:20]:
        rec_title = df.iloc[m_id]["display_title"]
        rec_score = similarity * 100  # Convert to percentage
        print(f"• {rec_title:<40} (Match: {rec_score:.1f}%)")


print("Building recommendation model, please wait...")

create_json()

df = pd.read_json("tmdb_dataset.json") #Read json file

create_ratings_file(df) #use cf.py file

create_movie_vectors_file() #use word2vec.py file

movie_vectors=np.load("movie_vectors.npy")

with open("ratings.pkl", "rb") as f:
    movie_voters, user_ratings, user_means = pickle.load(f)

overview_index,movieTF_IDF,movie_norms=build_overview_index(df)

movie_genres,genre_index,genre_weights,movie_weights=build_genre_index(df)

cast_index=build_cast_index(df)

calculate_statistics(df,overview_index,movieTF_IDF,movie_norms,movie_genres,genre_index,genre_weights,movie_weights,movie_voters,user_ratings,user_means,movie_vectors)

with open("statistics.pkl", "rb") as f:

    (overview_mean,genre_mean,cf_item_mean,word2vec_mean,overview_std,genre_std,cf_item_std,word2vec_std)=pickle.load(f)

print("Model ready!\n")

display_titles=df["display_title"].copy()

available_titles = display_titles.dropna().tolist()  # list of all available (non-empty) titles

while True:
    run_ui() #only works in terminal

