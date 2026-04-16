# uncertainty-DIR

In this file, you need to pay attention to serveral files:

1. train.py, which is the main file
2. splite _CP.py which is the splite conformal prediction file
3. network.py which defines the network of our model


Then, you need to:
    1. integrate the splite CP into the train.py
How?
    In the file : network.py, line 200, we defined a network module with one layer.
You need to : integrated this module with the main model （I do not remember if I have finished this part, pls double check it).
    Then, use the code in splite_CP.py to train the CP module (You need to think which gradient should be passed here?)
    Then, train the model with train set.
    |
    |----> feed the batch into the backbone, get the feature
    |----> pass the feature to two directions, 
                    |----> one is to predict label
                    |----> one is to build the prediction interval |----> split CP module  |----> get the CP loss
    |----> use the above two items to NLL