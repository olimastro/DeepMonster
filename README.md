# DeepMonster
Monstruous deep learning framework built on top of Theano and other frameworks such as blocks, fuel and lasagne.

The main motivation was that doing RNN and extending them in lasagne is a bit for now cumbersome. Adding batch norm
and converting their LSTM to a ConvLSTM meant basically to copy paste the whole class and hard code the changes. 

From fuel it uses pretty much all the data handling part
From blocks it uses the MainLoop class (extensions are heavily used!) and the Algorithm classes

The core of building a neural net is in adlf.

From lasagne it is pretty much a toolbox of added stuff (like batch norm and weight norm) when not using
adlf but going more for straight lasagne. 
