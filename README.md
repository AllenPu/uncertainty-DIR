# uncertainty-DIR

In this file, you need to pay attention to serveral files:

1. train.py, which is the main file
2. splite _CP.py which is the splite conformal prediction file
3. network.py which defines the network of our model


Updated list
Since the main code is based on CQR: model output-> y_pred, upper, lower, feature,
    we also need to impelement the split CP where we only use the **y_pred**

Then, I have acheived:
1. change the name of split_cp_loss -> coverage loss (pls revise the train.py code correspondingly)
    calibrate_qhat_from_batch() method is used to make estimation of qhat and misscoverage loss based on CQR
    calibrate_qhat_splitCP() method is used to make estimation of qhat and misscoverage loss based on Split_CP

2. add the cqr loss with pineball loss -> the orginal paper loss

3. gave you a demo on which part of network to update here:

    for main part you need to first build one optimizer:
        opt_extractor = optim.Adam(model.model_extractor.parameters(), lr=args.lr, weight_decay=5e-4)

    for the regressor we can have:
        opt_regressor = optim.Adam(model.pred_head.parameters(), lr=args.lr, weight_decay=5e-4)

    you need to new another optimizer with:
        opt_cp = optim.Adam(model.interval_head.parameters(), lr=args.lr, weight_decay=5e-4)

    Then, in per-batch training, when you estimate the q_hat, you can also have the loss (which I achieved already):
        first, update the parameter of the main model (extractor and pred_head) with nll_loss. 
        second, update the parameter of cp module with cp_loss. 