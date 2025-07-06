import torch


#
# return the precentile value from the vector
#
def percentile_excluding_index(vector, percentile):
        percentile_value = torch.quantile(vector, percentile)
        
        return percentile_value
#
# input is x and model itself
#
def get_scores(X, model):
    # revised to adapt to the model putput
    _, y_pred, _ = model(torch.tensor(X, dtype=torch.float32))
    scores = torch.nn.functional.softmax(y_pred, dim=1)
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


#
#
#
def find_intervals_above_value_with_interpolation(x_values, y_values, cutoff):
    intervals = []
    start_x = None
    if y_values[0] >= cutoff:
        start_x = x_values[0]
    for i in range(len(x_values) - 1):
        x1, x2 = x_values[i], x_values[i + 1]
        y1, y2 = y_values[i], y_values[i + 1]

        if min(y1, y2) <= cutoff < max(y1, y2):
            # Calculate the x-coordinate where the line crosses the cutoff value
            x_cross = x1 + (x2 - x1) * (cutoff - y1) / (y2 - y1)

            if x1 <= x_cross <= x2:
                if start_x is None:
                    start_x = x_cross
                else:
                    intervals.append((start_x, x_cross))
                    start_x = None

    # If the line ends above cutoff, add the last interval
    if start_x is not None:
        intervals.append((start_x, x_values[-1]))

    return intervals


#
# X is from the train (new input ones) and later is from val
#
def get_cp_lists(X, args, range_vals, X_cal, Y_cal, model):
    scores, all_scores = get_all_scores(range_vals, X_cal, Y_cal, model)
    pred_scores = get_scores(X, model)
    alpha = args.alpha

    percentile_val = percentile_excluding_index(all_scores, alpha)
        
    all_intervals = []
    for i in range(len(pred_scores)):
        all_intervals.append(find_intervals_above_value_with_interpolation(range_vals, pred_scores[i], percentile_val))

    return all_intervals

#
# get the interval for prediction on VAL
# range_vals is the list [group1, group2, group3,..., group N]
#
def get_intervals(X_test, X_cal, y_cal, args, range_vals, model):
        intervals = get_cp_lists(X_test, args, range_vals, X_cal, y_cal, model)
        #actual_intervals = invert_intervals(intervals)
        return intervals