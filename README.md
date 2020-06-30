# RNN--Movie-Review-Sentiment-Analysis

### Learning RNN for sequential data
RNNs are specialized neural-based approaches that are effective at processing sequential information. An RNN recursively applies a computation to every instance of an input sequence conditioned on the previous computed results. These sequences are typically represented by a fixed-size vector of tokens which are fed sequentially (one by one) to a recurrent unit. RNN have a memory which remembers all information about what has been calculated. It uses the same parameters for each input as it performs the same task on all the inputs or hidden layers to produce the output. This reduces the complexity of parameters, unlike other neural networks. 

![Architecture of Recurrent neural network for textual input data](https://github.com/SonaleeJ/RNN--Movie-Review-Sentiment-Analysis/blob/master/Rnn0.png)
###### Figure: Architecture of Recurrent neural network for textual input data

Neural network can process numeric inputs only; it can’t process text inputs. Hence textual data is encoded to numeric data. In RNN sequence of word is the input. Hence such words are converted into vectors. We use GloVe corpus for the vocabulary of word to support our model. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space (Source: https://nlp.stanford.edu/projects/glove/). To produce input sequence of contextual data we use word embedding. Word embedding is one of the most popular representation of document vocabulary. It is capable of capturing context of a word in a document, semantic and syntactic similarity, relation with other words, etc. These words are tokenized and converted into vector matrix of tokens using one hot encoding.__

In our implementation, model is built using RNN which consists of input layer, word embedding layer, LSTM layer (Long-short term memory unit layer), output layer (sigmoidal function is used for transfer function).__
Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (such as images, text), but also entire sequences of data. A common LSTM unit is composed of a cell, an input gate, an output gate and a forget gate. The cell remembers values over arbitrary time intervals and the three gates regulate the flow of information into and out of the cell. LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the exploding and vanishing gradient problems that can be encountered when training traditional RNNs.
Model is trained to learn the contextual data i.e. user review data to identify sentiment of user.<br />



#### Dataset: For implementation we will be using IMDB movie review dataset for sentiment analysis
IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

#### Columns:

•	Reviews: Textual data

•	Sentiment: positive/negative
(Source: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)


#### Steps to build the model for sentiment analysis:
1.	Load in and visualize the data
2.	Data Processing — convert to lower case
3.	Data Processing — Remove punctuation
4.	Data Processing — Create list of reviews
5.	Tokenize — Create Vocab to mapping dictionary
6.	Tokenize — Encode the words
7.	Tokenize — Encode the labels using one hot encoding
8.	Padding / Truncating the remaining data
9.	Word embedding and converting words into vector
10.	Training, Validation, Test Dataset Split
11.	Define the LSTM Network Architecture
12.	Training the Network
13.	Testing (on Test data and User- generated data)
 For RNN The layers are as follows:
1.	Tokenize : This is not a layer for RNN network but a mandatory step of converting our words into tokens (integers)
2.	Embedding Layer: that converts our word tokens (integers) into embedding of specific size
3.	LSTM Layer: defined by hidden state dims and number of layers
4.	Dense Layer: Densely connected hidden layers
5.	Sigmoid Activation Layer: that turns all output values in a value between 0 and 1
6.	Output: Sigmoid output from the last timestep is considered as the final output of this network

#### For RNN The layers are as follows:
1.	Tokenize : This is not a layer for RNN network but a mandatory step of converting our words into tokens (integers)
2.	Embedding Layer: that converts our word tokens (integers) into embedding of specific size
3.	LSTM Layer: defined by hidden state dims and number of layers
4.	Dense Layer: Densely connected hidden layers
5.	Sigmoid Activation Layer: that turns all output values in a value between 0 and 1
6.	Output: Sigmoid output from the last timestep is considered as the final output of this network 

Following figure shows the architecture of our model.

![Architecture of RNN for sentiment analysis](https://github.com/SonaleeJ/RNN--Movie-Review-Sentiment-Analysis/blob/master/Rnn.png)

##### Figure: Architecture of RNN for sentiment analysis

#### Conclusion:

Text classification is one of the most common natural language processing tasks. In our implementation we have done sentiment analysis using RNN with NLP. It needs large amount of data to build the corpus of words and to train the model. Model works well with 100 words of input and training of 6 epochs. To improve accuracy, we can add more layers in RNN and change the learning rate.

#### Observation:

•	The model’s test accuracy to identify the sentiment of viewer based on his/her review is 84%

•	As we can see in the experiment model needs only 6 epochs to learn the data. More than 6 epochs will overfit the model.<br />

Adding more layers in the network or using bi-LSTM improves the results. We can obtain accuracy above 90%.

