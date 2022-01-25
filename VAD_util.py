import numpy as np

def calculate_lambda(p_predicted_subgroup, p_predicted, calculation_space, logit_predicted_subgroup=None, logit_predicted=None):
    assert calculation_space in ['logit', 'probability']
    num_examples, num_group = p_predicted_subgroup.shape
    y_var = np.var(p_predicted)
    if calculation_space == 'logit':
        if logit_predicted_subgroup is None:
            p_predicted_subgroup = np.log(p_predicted_subgroup / (1 - p_predicted_subgroup))
        else:
            p_predicted_subgroup = logit_predicted_subgroup
        if logit_predicted is None:
            p_predicted = np.log(p_predicted / (1 - p_predicted))
        else:
            p_predicted = logit_predicted
        y_var = np.var(p_predicted)
    per_example_p_avg = np.mean(p_predicted_subgroup, axis=1)
    per_group_p_avg = np.mean(p_predicted_subgroup, axis=0)
    to_be_deducted = per_example_p_avg - np.mean(per_group_p_avg)
    value = p_predicted_subgroup - np.broadcast_to(per_group_p_avg, (num_examples, num_group)) - np.broadcast_to(to_be_deducted, (num_group, num_examples)).T
    prediction_variance = np.mean(value ** 2) * num_group / (num_group - 1)
    lambda_p = 1 - prediction_variance / y_var
    return lambda_p

def prediction_transformation(p_predicted, ind, lambda_p, calculation_space, p_mean_val, logit_mean_val, logit_predicted=None):
    assert calculation_space in ['logit', 'probability']
    if calculation_space == 'probability':
        return p_predicted[ind] * lambda_p + (1 - lambda_p) * p_mean_val
    elif calculation_space == 'logit':
        if logit_predicted is None:
            logit_predicted = np.log(p_predicted/(1-p_predicted))
        new_logit = logit_predicted * lambda_p + (1 - lambda_p) * logit_mean_val
        p_final = 1 / (1 + np.exp(-new_logit))
        return p_final[ind]
    else:
        raise NotImplementedError()