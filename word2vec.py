import numpy as np
from normalizations import sigmoid
import pandas as pd
import os

def stemming(word): #function used for stemming ( running=run, runs=run... )

    #special cases

    if len(word)<=4 or word.endswith(('ss','ee','is','us','ie')): #e.g boss, glass, pass...
        return word

    if word.endswith("ing") and len(word) - 3 >= 3:  # running=runn, transform it to run

        stemmed_word = word[:-3]  # only if final word length >=3

        if len(stemmed_word) >= 2 and stemmed_word[-1] == stemmed_word[-2]:
            stemmed_word = stemmed_word[:-1]

        return stemmed_word

    if word.endswith("ed") and len(word) - 2 >= 3:  # stopped=stopp, transform it to stop
        stemmed_word = word[:-2]  # only if final word length >=3
        if len(stemmed_word) >= 2 and stemmed_word[-1] == stemmed_word[-2]:
            stemmed_word = stemmed_word[:-1]

        return stemmed_word

    #suffixes sorted from biggest to smallest

    suffixes = (
        "ization", "ation", "ition", "ative",
        "ance", "ence", "ency", "able", "ible", "ious",
        "ment", "ness", "sion", "tion", "less", "ship",
        "ous", "ive", "ize", "est", "ful", "ly", "es", "s"
    )
    for suf in suffixes: #for each suffix

        if word.endswith(suf) and len(word)-len(suf)>=3: #if word ends with suffix

            stemmed_word=word[:-len(suf)] #only if final word length >=3

            return stemmed_word

    return word

def text_to_words(text,useStopWords,useStemming): #function that returns list, each word of a text is an element, e.g input: "The detective", output: ['the','detective']

    if useStopWords:

        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
            "t", "can", "will", "just", "don", "should", "now", "mr", "mrs", "miss", "dr", "sir"
        }

    else:
        stop_words = {}

    words=[]
    current_word=""
    for character in text.lower(): #for each character, transform to lowercase

        if character.isalpha(): #word has started

            current_word += character

        else: #word has ended

            if current_word and current_word not in stop_words:

                if useStemming:
                    words.append(stemming(current_word))
                else:
                    words.append(current_word)

            current_word="" #reset for next word

    if current_word and current_word not in stop_words: #if text ends with no punctuation, add last word

        if useStemming:
            words.append(stemming(current_word))
        else:
            words.append(current_word)

    return words

def corpus_to_words(corpus): #function that transforms corpus to list

    words=[]

    for text in corpus:

        words.append(text_to_words(text,useStopWords=False,useStemming=False))

    filtered_words=[]

    doc_freq = create_doc_freq(words)

    #if a word has frequency of 1, remove ( provides no informational value )

    for overview in words: #for each overview

        filtered_overviews = []

        for word in overview: #for each word in overview

            if doc_freq[word]>1:

                filtered_overviews.append(word)

        filtered_words.append(filtered_overviews)

    return filtered_words

def create_doc_freq(corpus): # dictionary that returns dictionary that holds frequency of each word in the document

    doc_freq = {}

    for text in corpus:  # for each text in corpus

        unique_words = set(text)  # create set for unique words, for each text(movie overview)

        for w in unique_words:  # for each word appearing, increment

            if w not in doc_freq:  # if a word in not in the dictionary
                doc_freq[w] = 1  # Set frequency as 1
            else:
                doc_freq[w] += 1  # if a word is in the dictionary, increment its frequency

    return doc_freq


def create_word_to_index(corpus_to_words): #function that returns dictionary {key: word, value:index}
    word_to_index={}

    for words in corpus_to_words:

        for word in words:

            if word not in word_to_index:

                word_to_index[word]=len(word_to_index) #so first word has value 0, second has 1, third has 2 . . .

    return word_to_index


def create_index_to_word(word_to_index): #function that returns dictionary {key: index, value:word}
    index_to_word={}
    for word,index in word_to_index.items():

         index_to_word[index]= word #reverse the already calculated word_to_index dictionary

    return index_to_word

def create_training_pairs(corpus_to_words,word_to_index,window_size): #function that creates pairs for words, e.g. (detective,investigation)

    #Use skip-gram method. For every word, create training pair according to window_size

    #e.g. (The detective investigate the case), if window_size=1, pairs for detective are: (detective, The), (detective, investigates)

    training_pairs=[]

    for words in corpus_to_words:

        n_words=len(words)

        for center_position in range(len(words)):

            center_word=words[center_position]
            center_index=word_to_index[center_word]

            start=max(0,center_position-window_size) #avoid minus position
            end=min(n_words,center_position+window_size+1) #avoid exceeding text size

            for context_position in range(start,end):

                if context_position!=center_position: #avoid creating pairs for same word

                    context_word=words[context_position]
                    context_index=word_to_index[context_word]

                    training_pairs.append((center_index,context_index))

    return np.array(training_pairs, dtype=np.int32)

def create_negative_samples_batch(center_indices,context_indices,vocab_size,num_negative): #function that creates pairs of words that don't appear together in our corpus (used for neural network training), e.g. (detective,banana)

    batch_size=len(center_indices)

    negative_samples=np.random.randint(0,vocab_size,size=(batch_size,num_negative)) #will return dimensions (1024 * 5)

    invalid=(negative_samples==center_indices[:,None]) | (negative_samples==context_indices[:,None]) #add dimension for correct comparison, avoid using the same word (center_indices) or a context word (context_indices)

    while np.any(invalid): #if at least 1 True exists (invalid sample)

        negative_samples[invalid]=np.random.randint(0, vocab_size,size=np.sum(invalid))  # generate new values for invalid samples, np.sum(invalid) = number of trues

        invalid = (negative_samples == center_indices[:, None]) | (negative_samples == context_indices[:, None])  # add dimension for correct comparison, avoid using the same word (center_indices) or a context word (context_indices)

    return negative_samples



def create_weights(vocab_size,embedding_size): #function that creates weights used for neural network training:W, W_out

    # #embedding_size = how many values the vector has for each word

    W=np.random.randn(vocab_size,embedding_size).astype(np.float32) * 0.01 #for when word is center
    W_out=np.random.randn(vocab_size,embedding_size).astype(np.float32) * 0.01  #for when word is context

    #multiply by 0.01 to avoid big starting weights

    return W,W_out

def predict_center_score(center_indices,context_indices,W,W_out): #function that calculates the score between a center word and its context words

    center_weights=W[center_indices]
    context_weights=W_out[context_indices]

    scores=np.sum(center_weights[:, None, :] * context_weights,axis=2 )

    #center_weights dimensions: (1024,50), context weights dimensions: (1024,6,50),
    # [:,None,:] adds dimension so center_weights become (1024,1,50),
    # use np.sum,axis=2, to calculate for each weight (dot product)

    probabilities=sigmoid(scores) #convert to [0,1] range

    return probabilities,center_weights,context_weights

def train_batch(center_indices,context_indices,labels,W,W_out,learning_rate): #function that trains weights for a batch

    probabilities,center_weights,context_weights=predict_center_score(center_indices,context_indices,W,W_out)

    errors=probabilities-labels #sigmoid error (derivative dLoss/dscore), label is true value (0 or 1)

    grad_center= np.sum(errors[:,:,None] *context_weights, axis=1) #derivative * weight (derivative dLoss/dcenter = dLoss/dscore * dscore/dcenter = error * context_weight)

    # errors dimensions: (1024,6)
    # context weights dimensions: (1024,6,10),
    # [:,:,None] adds dimension so errors become (1024,6,1)
    # use np.sum, axis=1 to calculate for each context

    grad_context=errors[:,:,None]*center_weights[:,None,:] #derivative * weight (derivative dLoss/dcontext = dLoss/dscore * dscore/dcontext = error * center_weight)

    # errors dimensions: (1024,6)
    # center_weights dimensions: (1024,50)
    # [:,:,None] adds dimension so errors become (1024,6,1)
    # [:,None,:] adds dimension so center_weights become (1024,6,50),
    # multiply each error by corresponding center vector
    #result (1024,6,50), one gradient vector for each context word

    np.add.at(W,center_indices,-learning_rate * grad_center) #use np.add.at in case same word appears 2 times or more in a batch

    for i in range(context_indices.shape[1]): #for each word in context_indices

        np.add.at(W_out,context_indices[:,i],-learning_rate * grad_context[:,i]) #use [:i] to take vector corresponding to current word

def train_word2vec(training_pairs,vocab_size,embedding_size,num_negative,learning_rate,epochs):

    W,W_out=create_weights(vocab_size,embedding_size) #initialize weights

    batch_size=1024

    for epoch in range(epochs):

        np.random.shuffle(training_pairs) #so training pairs appear in different order for each epoch

        for i in range(0,len(training_pairs),batch_size):

            batch=training_pairs[i:i+batch_size] #get 1024 training pairs at once

            center_indices=batch[:,0] #first element is center word, [:,0] to take all center words at once

            positive_contexts = batch[:, 1]  #second element is positive context word, [:,0] to take all context words at once

            negative_samples=create_negative_samples_batch(center_indices,positive_contexts ,vocab_size,num_negative) #negative samples

            context_indices=np.concatenate((positive_contexts[:,None],negative_samples),axis=1) #merge positive and negative indices

            labels = np.zeros( (len(batch), num_negative + 1),dtype=np.float32)  #set all labels as 0, use +1 since first label is positive (1 positive, 5 negatives)

            labels[:,0]=1  #first label is positive, set to 1

            train_batch(center_indices,context_indices,labels,W,W_out,learning_rate)  #train positive and negative pairs

    return W,W_out


def create_movie_vectors(corpus_to_words,word_to_index,W,movieTF_IDF): #for each movie, create a movie vectors combining each word vector, using TF_IDF weights

    movie_vectors=[]

    embedding_size=W.shape[1] #shape[1] in order to take columns

    for movie_index,words in enumerate(corpus_to_words): #for each movie overview

        word_vectors=[]
        TF_IDF_weights=[]

        if movieTF_IDF[movie_index]:
            currentTF_IDF=movieTF_IDF[movie_index]

        else: #In case TF_IDF unavailable for current movie
            currentTF_IDF={}

        for word in words: #for each word appearing in overview

            if word in word_to_index:

                if currentTF_IDF:
                    TF_IDF_weight=currentTF_IDF[word]

                else:
                    TF_IDF_weight=0.0

                word_index=word_to_index[word] #get index of word
                word_vectors.append(W[word_index] * TF_IDF_weight) #word's weight * TF_IDF weight
                #word_vectors.append(W[word_index])
                TF_IDF_weights.append(TF_IDF_weight)

        if word_vectors and np.sum(TF_IDF_weights)!=0:

            movie_vector=np.sum(word_vectors,axis=0)/np.sum(TF_IDF_weights) #axis=0 in order to take rows, weighted average
            #movie_vector=np.mean(word_vectors,axis=0)

        else:

            movie_vector=np.zeros(embedding_size)

        movie_vectors.append(movie_vector)

    return np.array(movie_vectors)

def find_movie_vectors_similarity(movie_index,movie_vectors): #function that returns Cosine Similarity between two movies, using the movie_vectors

    movie_vector=movie_vectors[movie_index] #movie_vector of current movie

    #Cosine Similarity (A * B / ||A|| * ||B|| )

    dot_products=np.dot(movie_vectors,movie_vector) #A * B
    movie_norm=np.linalg.norm(movie_vector) #||A||
    all_norms=np.linalg.norm(movie_vectors,axis=1) #||B||

    #all_norms shape (50.000 x 50), (movies x movie_embedding_size), use axis = 1 to take for each movie

    similarities=np.zeros(len(movie_vectors))

    denominator=movie_norm*all_norms #||A|| * ||B||
    np.divide(dot_products,denominator,out=similarities, where=denominator>0) #avoid division by 0, A * B / ||A|| * ||B||, save to similarities

    similarities[movie_index]=0 #movie shouldn't be compared with itself

    similarities={movie_id:score for movie_id,score in enumerate(similarities)} #convert to dictionary

    return similarities


def create_movie_TF_IDF(corpus_to_words):

    #Calculate DF (Document Frequency)

    doc_freq=create_doc_freq(corpus_to_words) #dictionary that holds frequency of each word in the document

    #Caculate TF-IDF (Term Frequency - Inverse Document Frequency)

    N=len(corpus_to_words) #number of total movies
    movie_TF_IDF=[] #TF_IDF for each movie

    for movie_id,text in enumerate(corpus_to_words): #for each movie and text in the corpus

        tf={} #dictionary that holds how many times a word appears in an overview

        for w in text: #for each word in an overview

            if w not in tf: #if a word is not in the dictionary
                tf[w]=1 #Set frequency as 1
            else:
                tf[w]+=1 #if a word is in the dictionary, increment its frequency

        TF_IDF_dict={} #each movie has its own TF-IDF dictionary

        for word,count in tf.items(): #for each word and count in TF

            #TF-IDF= TF * IDF
            #TF: how many times a word appears in a certain movie overview
            #IDF: takes into account how many times a word appears in the entire document
            #IDF = log( Number of different overviews (movies) / times a word appears in the entire document)

            if doc_freq[word] > 1:  # if a word appears only one time in the document, remove it (no use)

                TF=count
                IDF=np.log(N/(doc_freq[word]))

                TF_IDF=TF*IDF
                TF_IDF_dict[word]=TF_IDF #store TF_IDF value for each word

        movie_TF_IDF.append(TF_IDF_dict) #store TF_IDF for each movie

    return movie_TF_IDF


def calculate_weights():

    print("Building word2vec Neural Network...")

    df = pd.read_json("tmdb_dataset.json")  # Read json file

    corpus = df["overview"].fillna("").tolist()  # fillna("") in case of empty overview

    words = corpus_to_words(corpus)

    word_to_index = create_word_to_index(words)

    train_pairs=create_training_pairs(words,word_to_index,window_size=6)

    vocab_size=len(word_to_index)

    W,W_out=train_word2vec(train_pairs,vocab_size,embedding_size=50,num_negative=5,learning_rate=0.0025,epochs=200)

    return words,word_to_index,W

def create_movie_vectors_file():

    if os.path.exists("movie_vectors.npy"): #if file already exists, return
        return

    words,word_to_index,W=calculate_weights()

    word2vec_TF_IDF=create_movie_TF_IDF(words)

    movie_vectors = create_movie_vectors(words, word_to_index, W, word2vec_TF_IDF)

    np.save("movie_vectors.npy", movie_vectors)

    print("movie_vectors.npy successfully created")













