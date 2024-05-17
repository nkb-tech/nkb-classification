from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor, nn

DEFAULT_FOCAL_GAMMA = 2.0


class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = DEFAULT_FOCAL_GAMMA,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".'
            )

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = x.log_softmax(dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

# class MultitaskLoss:
#     def __init__(self, loss):
#         self.loss = loss
#         self.iteration = -1

#     def backward(self):
#         self.loss['loss'].backward()

#     def item(self):
#         return self.loss['loss'].item()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         self.iteration += 1
#         if self.iteration < len(self.loss):
#             return self.loss.values()[self.iteration]
#         raise StopIteration

class MultitasCriterion:
    """
    Wrapper for any loss function, which
    allows it to work with a multi-task classification.
    When called, it returns a sum of losses for each task.
    """
    def __init__(self, criterion, device):
        self.criterion = criterion
        self.device = device

        self.criterion.to(device)

    def __call__(self,
                 pred: dict,
                 true: dict):
                #  running_loss: defaultdict = None,
                #  should_return_separate_loss: bool = True):
        """
        Computes overall loss over tasks.

        In case of multitask classification, predictions
        and ground true values are dictionaries, in which
        keys are task names.

        Parameters
        ----------
        pred : dict
            Model predicted labels for each task.
        true : dict
            Ground true labels for each task.
        should_return_separate_loss : bool
            Whether to return separate loss for each task.
        """
        assert pred.keys() == true.keys()
        loss = 0

        separate_loss = defaultdict()

        for target_name in pred.keys():
            target_loss = self.criterion(
                pred[target_name], true[target_name].to(self.device)
            )
            # if running_loss is not None:
                # running_loss[target_name].append(
                    # target_loss.item()
                # )
            separate_loss[target_name] = target_loss#.item()

            loss += target_loss

        # if should_return_separate_loss:
        #     return loss, separate_loss
        separate_loss['loss'] = loss
        return separate_loss
        # if running_loss is not None:
        #     return loss, running_loss
        # else:
        #     return loss

def get_loss(cfg_loss, device):
    if cfg_loss["type"] == "CrossEntropyLoss":
        weight = None
        if "weight" in cfg_loss:
            weight = torch.tensor(cfg_loss["weight"], dtype=torch.float)
        loss = nn.CrossEntropyLoss(weight).to(device)
        # return nn.CrossEntropyLoss(weight).to(device)
    elif cfg_loss["type"] == "FocalLoss":
        alpha = None
        if "alpha" in cfg_loss:
            alpha = torch.tensor(cfg_loss["alpha"], dtype=torch.float)
        gamma = DEFAULT_FOCAL_GAMMA
        if "gamma" in cfg_loss:
            gamma = cfg_loss["gamma"]
        loss = FocalLoss(alpha, gamma).to(device)
        # return FocalLoss(alpha, gamma).to(device)
    else:
        raise NotImplementedError(
            f'Unknown loss type in config: {cfg_loss["type"]}'
        )

    if cfg_loss['task'] == 'multi':
        return MultitasCriterion(loss, device)
    else:
        return loss
