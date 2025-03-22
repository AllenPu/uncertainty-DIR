import torch

#
# input is x and model itself
#
def get_scores(X, model):
    scores = torch.nn.functional.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1)
    return scores

#
# range_vals is a linspace from min y to max y
#
def get_all_scores(range_vals, x, y, model):
    step_val = (max(range_vals) - min(range_vals))/(len(range_vals) - 1)
    indices_up = torch.ceil((y - min(range_vals))/step_val).squeeze()
    indices_down = torch.floor((y - min(range_vals))/step_val).squeeze()
    
    how_much_each_direction = ((y.squeeze() - min(range_vals))/step_val - indices_down)

    weight_up = how_much_each_direction
    weight_down = 1 - how_much_each_direction

    bad_indices = torch.where(torch.logical_or(y.squeeze() > max(range_vals), y.squeeze() < min(range_vals)))
    indices_up[bad_indices] = 0
    indices_down[bad_indices] = 0
    
    scores = get_scores(x, model)
    all_scores = scores[torch.arange(len(x)), indices_up.long()] * weight_up + scores[torch.arange(len(x)), indices_down.long()] * weight_down
    all_scores[bad_indices] = 0
    return scores, all_scores
