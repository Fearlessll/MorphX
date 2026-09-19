"""Survival losses and evaluation helpers."""

import math

import numpy as np
import torch
import torch.nn as nn


def cox_loss(
    survtime: torch.Tensor,
    event: torch.Tensor,
    hazard_pred: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """Compute the negative Cox partial log-likelihood.

    Ties use the Breslow risk-set convention. The negative log likelihood is
    averaged over observed events. ``logcumsumexp`` keeps the calculation
    stable without explicitly exponentiating risk scores.
    """

    del device
    times = torch.as_tensor(survtime, device=hazard_pred.device, dtype=torch.float32).reshape(-1)
    events = torch.as_tensor(event, device=hazard_pred.device, dtype=torch.float32).reshape(-1)
    scores = hazard_pred.to(dtype=torch.float32).reshape(-1)
    if not (times.numel() == events.numel() == scores.numel()):
        raise ValueError("survtime, event, and hazard_pred must have the same number of samples.")
    if scores.numel() == 0:
        raise ValueError("Cox loss cannot be computed on an empty batch.")
    order = torch.argsort(times, descending=True)
    sorted_times = times[order]
    sorted_scores = scores[order]
    sorted_events = events[order]
    log_risk = torch.logcumsumexp(sorted_scores, dim=0)
    _, group_counts = torch.unique_consecutive(sorted_times, return_counts=True)
    group_ends = torch.cumsum(group_counts, dim=0) - 1
    group_starts = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=group_ends.device), group_ends[:-1] + 1]
    )
    total_events = sorted_events.sum()
    if total_events <= 0:
        return scores.sum() * 0.0

    # Breslow ties: all events at one time share the full risk set at that time.
    numerator = scores.sum() * 0.0
    denominator = scores.sum() * 0.0
    for start, end in zip(group_starts, group_ends):
        start_i, end_i = int(start.item()), int(end.item())
        group_events = sorted_events[start_i : end_i + 1]
        event_count = group_events.sum()
        if event_count > 0:
            numerator = numerator + (
                sorted_scores[start_i : end_i + 1] * group_events
            ).sum()
            denominator = denominator + event_count * log_risk[end_i]
    return (denominator - numerator) / total_events


def modified_cox_loss(survtime, event, hazard_pred, device=None):
    """Historical two-term MorphX loss from both source training scripts.

    The source adds a standard row-wise risk-set term and a second column-wise
    term, each averaged over the whole batch. This differs from the manuscript's
    displayed Cox partial-likelihood equation. It is retained for reproducing
    source-code training behavior; use ``cox_loss`` for the standard variant.
    """

    del device
    times = torch.as_tensor(survtime, device=hazard_pred.device, dtype=torch.float32).reshape(-1)
    events = torch.as_tensor(event, device=hazard_pred.device, dtype=torch.float32).reshape(-1)
    scores = hazard_pred.to(dtype=torch.float32).reshape(-1)
    if not (len(times) == len(events) == len(scores)) or len(scores) == 0:
        raise ValueError("Survival times, events, and scores must have the same positive length")
    risk_set = times[None, :] >= times[:, None]
    row_log = torch.logsumexp(scores[None, :].expand(len(scores), -1).masked_fill(~risk_set, -torch.inf), dim=1)
    col_log = torch.logsumexp(scores[:, None].expand(-1, len(scores)).masked_fill(~risk_set, -torch.inf), dim=0)
    return -torch.mean((scores - row_log) * events) - torch.mean((scores - col_log) * events)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return preds.eq(labels).float().mean()


def accuracy_cox(hazardsdata, labels):
    hazards = np.asarray(hazardsdata).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if hazards.size == 0:
        return float("nan")
    return float(((hazards > np.median(hazards)).astype(int) == labels).mean())


def cox_log_rank(hazardsdata, labels, survtime_all):
    """Return a log-rank p-value, or NaN when a split is not estimable."""

    hazards = np.asarray(hazardsdata).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    times = np.asarray(survtime_all).reshape(-1)
    groups = hazards > np.median(hazards)
    if len(hazards) < 4 or np.unique(groups).size < 2:
        return float("nan")
    try:
        from lifelines.statistics import logrank_test

        result = logrank_test(
            times[~groups],
            times[groups],
            event_observed_A=labels[~groups],
            event_observed_B=labels[groups],
        )
        return float(result.p_value)
    except ModuleNotFoundError:
        # Keep training/evaluation usable for the lightweight CPU demo when
        # the optional lifelines package is not installed.
        from scipy.stats import chi2

        valid = np.isfinite(times) & np.isfinite(labels) & np.isfinite(hazards)
        times = times[valid]
        labels = labels[valid].astype(int)
        groups = groups[valid]
        observed = expected = variance = 0.0
        for event_time in np.unique(times[labels == 1]):
            at_risk = times >= event_time
            at_time = (times == event_time) & (labels == 1)
            n_total = int(at_risk.sum())
            n_group = int((at_risk & groups).sum())
            event_count = int(at_time.sum())
            group_events = int((at_time & groups).sum())
            if n_total <= 1:
                continue
            observed += group_events
            expected += event_count * n_group / n_total
            variance += (
                n_group
                * (n_total - n_group)
                * event_count
                * (n_total - event_count)
                / (n_total * n_total * (n_total - 1))
            )
        if variance <= 0:
            return float("nan")
        return float(chi2.sf((observed - expected) ** 2 / variance, 1))


def cindex_lifeline(hazards, labels, survtime_all):
    hazards = np.asarray(hazards).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    times = np.asarray(survtime_all).reshape(-1)
    if len(hazards) < 2:
        return float("nan")
    if not (len(hazards) == len(labels) == len(times)):
        raise ValueError("hazards, labels, and survival times must have equal lengths.")
    try:
        from lifelines.utils import concordance_index

        return float(concordance_index(times, -hazards, labels))
    except ModuleNotFoundError:
        valid = np.isfinite(hazards) & np.isfinite(labels) & np.isfinite(times)
        hazards = hazards[valid]
        labels = labels[valid].astype(int)
        times = times[valid]
        permissible = (labels[:, None] == 1) & (times[:, None] < times[None, :])
        denominator = int(permissible.sum())
        if denominator == 0:
            return float("nan")
        concordant = ((hazards[:, None] > hazards[None, :]) & permissible).sum()
        tied = ((hazards[:, None] == hazards[None, :]) & permissible).sum()
        return float((concordant + 0.5 * tied) / denominator)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def init_max_weights(module):
    for child in module.modules():
        if isinstance(child, nn.Linear):
            stdv = 1.0 / math.sqrt(child.weight.size(1))
            child.weight.data.normal_(0, stdv)
            if child.bias is not None:
                child.bias.data.zero_()


def dfs_freeze(model):
    for child in model.children():
        for parameter in child.parameters():
            parameter.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for child in model.children():
        for parameter in child.parameters():
            parameter.requires_grad = True
        dfs_unfreeze(child)
