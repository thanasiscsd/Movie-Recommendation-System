import numpy as np

def standardization(scores,mean,std): #z score for sigmoid input

    if std==0:
        return 0.0

    return (scores-mean)/std

def sigmoid(z_scores):
    return 1 / (1 + np.exp(-z_scores))

def minmax_normalization(scores):

    #check only movies with score>0.0

    filtered_scores={movie_id: score for movie_id, score in scores.items() if score>0.0}

    if not filtered_scores:

        return {}

    min_score = min(filtered_scores.values())
    max_score = max(filtered_scores.values())

    normalized={} #dictionary which will hold ranked movies

    for movie_id,score in filtered_scores.items(): #for each rank and id in a movie
        normalized[movie_id]=(score-min_score)/(max_score-min_score)

    return normalized

def rank_normalization(scores):

    #check only movies with score>0.0

    filtered_scores={movie_id: score for movie_id, score in scores.items() if score>0.0}

    #sort movie ids according to their score (ascending)
    sorted_movies = sorted(filtered_scores.keys(), key=lambda x: filtered_scores[x])

    N=len(sorted_movies)
    ranked={} #dictionary which will hold ranked movies

    for rank,movie_id in enumerate(sorted_movies): #for each rank and id in a movie

        ranked[movie_id]=rank/(N-1) #best rank is rank N-1, N-1/(N-1)=1 (best scenario), worst ranke is 0, 0(N-1)=0 (worst scenario)

    return ranked