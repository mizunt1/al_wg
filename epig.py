import math
import torch
from torch import Tensor
from typing import Sequence
"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

def check(
    scores: Tensor, min_value: float = 0.0, max_value: float = math.inf, score_type: str = ""
) -> Tensor:
    """
    Warn if any element of scores is a NaN or lies outside the range [min_value, max_value].
    """
    epsilon = 10 * torch.finfo(scores.dtype).eps

    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        print(
            f"Invalid score (type = {score_type}, min = {min_score}, max = {max_score})"
        )

    return scores

def logmeanexp(x: Tensor, dim =None , keepdim = False):
    """
    Numerically stable implementation of log(mean(exp(x))).
    """
    if dim is None:
        size = math.prod(x.shape)
        return torch.logsumexp(x) - math.log(size)
    elif isinstance(dim, int):
        size = x.shape[dim]
        return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(size)
    elif isinstance(dim, (list, tuple)):
        size = math.prod(x.shape[d] for d in dim)
        return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(size)
    else:
        raise TypeError(f"Unsupported type: {type(dim)}")


def batch_epig(log_probs_pool, log_probs_target):
    to_fill = torch.zeros(len(log_probs_pool)) #[N_p]
    done = 0
    batch_size = 1000
    batches = math.ceil(len(log_probs_pool) / batch_size)
    for i in range(batches):
        if i == (batches-1):
            # last batch
            batch_size = len(log_probs_pool) - done
        log_probs_pool_batch = log_probs_pool[done: done+batch_size, :,:]
        score = conditional_epig_from_logprobs(log_probs_pool_batch, log_probs_target)
        score_mean_over_targets = torch.mean(score, dim=1)
        to_fill[done:done + batch_size] = score_mean_over_targets
        done += batch_size
    return to_fill

def conditional_epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    _, _, Cl = logprobs_pool.shape

    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    logprobs_targ = logprobs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    logprobs_joint = logprobs_pool + logprobs_targ  # [N_p, N_t, K, Cl, Cl]
    logprobs_joint = logmeanexp(logprobs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=2)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    log_term = logprobs_joint - logprobs_joint_indep  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(torch.exp(logprobs_joint) * log_term, dim=(-2, -1))  # [N_p, N_t]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p, N_t]
    
    return scores  # [N_p, N_t]
