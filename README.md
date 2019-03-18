# POS

![pos_png](https://raw.githubusercontent.com/Aniruddha-Tapas/how-to-use-syntaxnet/master/syntaxnet.gif)


## 1. What is speech tagging?

In corpus linguistics, part-of-speech tagging (POS tagging or PoS tagging or POST), also called grammatical tagging or word-category classificstion, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition and its context i.e., its relationship with adjacent and related words in a phrase, sentence, or paragraph. A simplified form of this is commonly taught to school-age children, in the identification of words as nouns, verbs, adjectives, adverbs, etc.





## 2. The Brown Corpus

The Brown Corpus of Standard American English was the first of the modern, computer readable, general corpora. It was compiled by W.N. Francis and H. Kucera, Brown University, Providence, RI. The corpus consists of one million words of American English texts printed in 1961. The texts for the corpus were sampled from 15 different text categories to make the corpus a good standard reference. Today, this corpus is considered small, and slightly dated. The corpus is, however, still used. Much of its usefulness lies in the fact that the Brown corpus lay-out has been copied by other corpus compilers. The LOB corpus (British English) and the Kolhapur Corpus (Indian English) are two examples of corpora made to match the Brown corpus. The availability of corpora which are so similar in structure is a valuable resourse for researchers interested in comparing different language varieties.
The Brown corpus consists of 500 texts, each consisting of just over 2,000 words. The texts were sampled from 15 different text categories. The number of texts in each category varies. [Read more...](https://www.sketchengine.eu/brown-corpus/)

## 3. Various Method to Execute POS


1. Markov Model
2. Hidden Markov Model
3. Viterbi algorithm
4. Baum–Welch algorithm


### 3.1 Markov Model

The HMM is based on augmenting the Markov chain. A Markov chain is a model
that tells us something about the probabilities of sequences of random variables,
states, each of which can take on values from some set. These sets can be words, or
tags, or symbols representing anything, like the weather. A Markov chain makes a
very strong assumption that if we want to predict the future in the sequence, all that matters is the current state. The states before the current state have no impact on the future except via the current state. It’s as if to predict tomorrow’s weather you could examine today’s weather but you weren’t allowed to look at yesterday’s weather.
Consider a sequence of state variables q1,q2,...,qi
. A Markov
model embodies the Markov assumption on the probabilities of this sequence: that
when predicting the future, the past doesn’t matter, only the present. Markov Assumption:
                
          P(qi = a|q1...qi−1) = P(qi = a|qi−1) 
          
<img src="https://cdn-images-1.medium.com/max/1600/1*zP2ofsYNmYlZL5FzR_SYzw.png"/>

Above figure shows a Markov chain for assigning a probability to a sequence of
weather events, for which the vocabulary consists of HOT, COLD, and WARM. The
states are represented as nodes in the graph, and the transitions, with their probabilities, as edges. The transitions are probabilities: the values of arcs leaving a given state must sum to 1.

Formally, a Markov chain is specified by the following components:
    
```
Q = q1q2 ...qN a set of N states
A = a11a12 ...an1 ...ann a transition probability matrix A, each aij                     representing the probability of moving from state i to state j
π = π1,π2,...,πN an initial probability distribution over states. πi
    is the probability that the Markov chain will start in state i. Some states         j may have πj = 0, meaning that they cannot be initial states.
    
    
```




### 3.2 Hidden Markov Model

In many cases, however, the events we are interested in are **hidden**: we don’t observe them directly. For example we don’t normally observe part-of-speech tags in a text. Rather, we see words, and must infer the tags from the word sequence. We call the tags hidden because they are not observed.
A hidden Markov model (HMM) allows us to talk about both observed events Hidden
Markov model (like words that we see in the input) and hidden events (like part-of-speech tags) that we think of as causal factors in our probabilistic model. An HMM is specified by the following components:


```
Q = q1q2 ...qN a set of N states
A = a11 ...ai j ...aNN a transition probability matrix A, each ai j representing the     probability of moving from state i to state 
O = o1o2 ...oT a sequence of T observations, each one drawn from a vocabulary
V =v1, v2,..., vn
B = bi(ot) a sequence of observation likelihoods, also called emission probabilities, each expressing the probability of an observation ot being generated
from a state i
π = π1,π2,...,πN an initial probability distribution over states. πi
    is the probability that the Markov chain will start in state i. Some states jmay have πj = 0, meaning that they cannot be initial states
```

</br>
</br>

### 3.3 Viterbi algorithm


Let Q(t,s) be the most probable sequence of hidden states of length t that finishes in the state s and generates o1,o2,o3...on. Let q(t,s) be the probability of this sequence.
then,
        q(t,s) = max(q(t,s') * p(s|s') * p(o[t]|s))
        
 where,
     p(s|s') is the transition probability,
     p(o[t]|s) is the output probability
 
 Note : Q(t,s) can be ddeternimed by remembering the argmax
     


### 3.4 Baum–Welch algorithm

In electrical engineering, computer science, statistical computing and bioinformatics, the Baum–Welch algorithm is used to find the unknown parameters of a hidden Markov model (HMM). It makes use of a forward-backward algorithm.

The Baum–Welch algorithm uses the well known EM algorithm to find the maximum likelihood estimate of the parameters of a hidden Markov model given a set of observed feature vectors.

## 4. How POS uses these algorithm

First we have to provide data in terms of simple plain text which will then converted into tokens. Also we have feed a hash mapping which contains information of a token_id and type of part of speech of that perticular token.We also have to add some fake token like start,stop,PUNC or UNK which will help model to think that if we want to start a sentence what kind of words we can use to start it.

start : Token which indicates the starting of the sentence
stop  : Token which indicates the stop/end of the sentence
UNK : Words that are rare or the words that occurs less frequently in our corpus


Now to generate a sentence we have to calculate the probability of a sentence starting with noun, pronoun, adjective, determiner, verb, adverb, preposition, conjunction, and interjection.

Then after selecting a perticular part of speech we have to calculate the probability of sentence strating with a word belonging to that perticular part of speech.

After selecting a starting word, now we have to calculate what type of future part of speech comes after the present part of speech, then which word will occur belonging to that part of speech.

And the process goes on...


## 5. Main Application

Part-of-Speech tagging is a well-known task in Natural Language Processing. It refers to the process of classifying words into their parts of speech (also known as words classes or lexical categories). This is a supervised learning approach.
It is mainly used to generate text.


For more depth understanding please watch thiv video:

<iframe width="916" height="515" src="https://www.youtube.com/embed/yE0dcDNRZjw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
 &copy; SerajRaval 




## 6. Code


### 1.
    sent = """We will meet at eight o'clock on Thursday morning."""
    text = Text(sent)
    text.pos_tags
    
output:
    
    [('We', 'PRON'),
     ('will', 'AUX'),
     ('meet', 'VERB'),
     ('at', 'ADP'),
     ('eight', 'NUM'),
     ("o'clock", 'NOUN'),
     ('on', 'ADP'),
     ('Thursday', 'PROPN'),
     ('morning', 'NOUN'),
     ('.', 'PUNCT')]
     
     
![pos](https://iq.opengenus.org/content/images/2019/02/pos.png)



### 2. 

    from numpy import array
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Embedding

    #generate a sequence from the model
    def generate_seq(model, tokenizer, seed_text, n_words):
        in_text, result = seed_text, seed_text
        # generate a fixed number of words
        for _ in range(n_words):
            # encode the text as integer
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            encoded = array(encoded)
            # predict a word in the vocabulary
            yhat = model.predict_classes(encoded, verbose=0)
            # map predicted word index to word
            out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text, result = out_word, result + ' ' + out_word
	return result

    #source text
    data = """ Jack and Jill went up the hill\n
            To fetch a pail of water\n
            Jack fell down and broke his crown\n
            And Jill came tumbling after\n """
    #integer encode text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([data])
    encoded = tokenizer.texts_to_sequences([data])[0]
    #determine the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    #create word -> word sequences
    sequences = list()
    for i in range(1, len(encoded)):
        sequence = encoded[i-1:i+1]
        sequences.append(sequence)
    print('Total Sequences: %d' % len(sequences))
    #split into X and y elements
    sequences = array(sequences)
    X, y = sequences[:,0],sequences[:,1]
    #one hot encode outputs
    y = to_categorical(y, num_classes=vocab_size)
    #define model
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=1))
    model.add(LSTM(50))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    #compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #fit network
    model.fit(X, y, epochs=500, verbose=2)
    #evaluate
    print(generate_seq(model, tokenizer, 'Jack', 6))
    
OUTPUT:
     
     _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding_1 (Embedding)      (None, 1, 10)             220
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 50)                12200
    _________________________________________________________________
    dense_1 (Dense)              (None, 22)                1122
    =================================================================
    Total params: 13,542
    Trainable params: 13,542
    Non-trainable params: 0
    _________________________________________________________________



    Epoch 496/500
        0s - loss: 0.2358 - acc: 0.8750
    Epoch 497/500
        0s - loss: 0.2355 - acc: 0.8750
    Epoch 498/500
        0s - loss: 0.2352 - acc: 0.8750
    Epoch 499/500
        0s - loss: 0.2349 - acc: 0.8750
    Epoch 500/500
        0s - loss: 0.2346 - acc: 0.8750




## Reference:
1. [O'REILLY](https://www.oreilly.com/library/view/mastering-machine-learning/9781788621113/51872ade-92b3-4149-afb7-b59665cbc0da.xhtml)
2. Wikipedia
3. [Medium](https://becominghuman.ai/part-of-speech-tagging-tutorial-with-the-keras-deep-learning-library-d7f93fa05537)
4. Machine Learning Mastry.
