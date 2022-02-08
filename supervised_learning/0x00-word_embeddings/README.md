# Word Embeddings

## Resources
* [An Introduction to Word Embeddings](https://www.springboard.com/blog/data-science/introduction-word-embeddings/ "An Introduction to Word Embeddings")
* [Introduction to Word Embeddings](http://hunterheidenreich.com/blog/intro-to-word-embeddings/ "Introduction to Word Embeddings")
* [Bag of Words](https://www.youtube.com/watch?v=IKgBLTeQQL8&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=6 "Bag of Words")
* [TF-IDF](https://www.youtube.com/watch?v=D2V1okCEsiE&list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm&index=8 "TF-IDF")

## References
* [`sklearn.feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#sklearn.feature_extraction.text.CountVectorizer "sklearn.feature_extraction.text.CountVectorizer")
* [`sklearn.feature_extraction.text.TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer "sklearn.feature_extraction.text.TfidfVectorizer")
* [`gensim.models.Word2Vec.__init__`](https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.__init__.html "gensim.models.Word2Vec.__init__")

## Tasks
### [0. Bag Of Words](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/0-bag_of_words.py "0. Bag Of Words")


Creates a bag of words embedding matrix.

---
### [1. TF-IDF](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/1-tf_idf.py "1. TF-IDF")

Creates a TF-IDF embedding.

---
### [2. Train Word2Vec](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/2-word2vec.py "2. Train Word2Vec")

Creates and trains a `genism` `word2vec` model.

---
### [3. Extract Word2Vec](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/3-gensim_to_keras.py "3. Extract Word2Vec")

Converts a `genism` `word2vec` model to a `keras` embedding layer.

---
### [4. FastText](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/4-fasttext.py "4. FastText")

Creates and trains a `genism` `fastText` model.

---
### [5. ELMo](https://github.com/kyle-gross/holbertonschool-machine_learning/blob/main/supervised_learning/0x00-word_embeddings/5-elmo "5. ELMo")

Text file with an answer to the multiple choice question:

When training an ELMo embedding model, you are training:

1. The internal weights of the BiLSTM
2. The character embedding layer
3. The weights applied to the hidden states

A. 1, 2, 3

B. 1, 2

C. 2, 3

D. 1, 3

E. 1

F. 2

G. 3

H. None of the above
