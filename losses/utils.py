import math
from typing import Optional, List, Union, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable
from torch.cuda.amp import autocast


@autocast(enabled=True, dtype=torch.float32)  # avoid numerical instability
def sinkhorn(
    a: Tensor,
    b: Tensor,
    C: Tensor,
    reg: float = 1e-1,
    maxIter: int = 1000,
    stopThr: float = 1e-9,
    verbose: bool = False,
    log: bool = True,
    eval_freq: int = 10,
    print_freq: int = 200,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t.    \gamma 1 = a
                \gamma^T 1= b
                \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------
    """
    M_EPS = 1e-16
    device = a.device
    na, nb = C.shape

    # a = a.view(-1, 1)
    # b = b.view(-1, 1)

    assert na >= 1 and nb >= 1, f"C needs to be 2d. Found C.shape = {C.shape}"
    assert na == a.shape[0] and nb == b.shape[0], f"Shape of a ({a.shape}) or b ({b.shape}) does not match that of C ({C.shape})"
    assert reg > 0, f"reg should be greater than 0. Found reg = {reg}"
    assert a.min() >= 0. and b.min() >= 0., f"Elements in a and b should be nonnegative. Found a.min() = {a.min()}, b.min() = {b.min()}"

    if log:
        log = {"err": []}

    u = torch.ones((na), dtype=a.dtype).to(device) / na
    v = torch.ones((nb), dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        # torch.matmul(u, K, out=KTu)
        KTu = torch.matmul(u.view(1, -1), K).view(-1)
        v = torch.div(b, KTu + M_EPS)
        # torch.matmul(K, v, out=Kv)
        Kv = torch.matmul(K, v.view(-1, 1)).view(-1)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print("Warning: numerical errors at iteration", it)
            u, v = upre, vpre
            break

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = (torch.matmul(u.view(1, -1), K) * v.view(1, -1)).view(-1)
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()
            log["err"].append(err)

        if verbose and it % print_freq == 0:
            print("iteration {:5d}, constraint error {:5e}".format(it, err))

        it += 1

    if log:
        log["u"] = u
        log["v"] = v
        log["alpha"] = reg * torch.log(u + M_EPS)
        log["beta"] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


def _reshape_density(density: Tensor, reduction: int) -> Tensor:
    assert len(density.shape) == 4, f"Expected 4D (B, 1, H, W) tensor, got {density.shape}"
    assert density.shape[1] == 1, f"Expected 1 channel, got {density.shape[1]}"
    assert density.shape[2] % reduction == 0, f"Expected height to be divisible by {reduction}, got {density.shape[2]}"
    assert density.shape[3] % reduction == 0, f"Expected width to be divisible by {reduction}, got {density.shape[3]}"
    return density.reshape(density.shape[0], 1, density.shape[2] // reduction, reduction, density.shape[3] // reduction, reduction).sum(dim=(-1, -3))

def dice_loss(inputs: Tensor, targets: Tensor, num_boxes: int) -> float:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes



def sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    num_boxes: int,
    alpha: Optional[float] = 0.25,
    gamma: float = 2
) -> Tensor:
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class Gaussianlayer(nn.Module):
    def __init__(self, sigma: Optional[List[int]] = None, kernel_size: int = 15) -> None:
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def forward(self, dotmaps: Tensor) -> Tensor:
        denmaps = self.gaussian(dotmaps)
        return denmaps


class Gaussian(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        sigmalist: List[int],
        kernel_size: int = 64,
        stride: int = 1,
        padding: int = 0,
        froze: bool = True
    ) -> None:
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)

        if froze: self.frozePara()

    def forward(self, dotmaps: Tensor) -> Tensor:
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps

    def frozePara(self) -> None:
        for para in self.parameters():
            para.requires_grad = False







