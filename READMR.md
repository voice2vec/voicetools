# Voice vectorization with theano and lasagne

### Features

1. Voice activity detection

There are some realization of VAD algoritms.  Zero Crossing rate, Energy, Magnitude realizations throe Theano library. 
Using [this](http://research.iaun.ac.ir/pd/mahmoodian/pdfs/UploadFile_2643.pdf) article for algoritms.

2. Vectorization

Vectorization of voice using lasagne. With voice vectors we can do something intresting like predict gender, accent or
to compare two voices.

3. Service

We want to do service with API for predicting voice vector.
