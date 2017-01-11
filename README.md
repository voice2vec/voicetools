# Voice vectorization with theano and lasagne

### Features

1. Voice activity detection

There are some realization of VAD algoritms.  Zero Crossing rate, Energy, Magnitude realizations through Theano library. 
Using [this](http://research.iaun.ac.ir/pd/mahmoodian/pdfs/UploadFile_2643.pdf) article for algoritms.

2. Vectorization

Vectorization of voice using lasagne. With voice vectors we can do something intresting... Predict gender, accent or to compare voices.

3. Service

We want to do service with API for predicting voice vector.

### Score

AUC score on our dataset with task 'same or not same voices': 0.8923 Dataset includes 51k voices with metainformation about authors.

### For what?

* Voice authorization
* Personalization voice servoces
* Clastering audio files
