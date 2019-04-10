import torch

from harmonic import AddSine


def log_weights_norm(gain=1.):
    def f(w):
        w[w < 2] = 2.
        w = gain / torch.log(w)
        return w

    return f


def pow_weights_norm(gain=1., power=3.):
    def f(w):
        w = gain / torch.pow(w, 1. / power)
        return w

    return f


def linear_weights_norm(gain=1.):
    def f(w):
        w = gain / w
        return w

    return f


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def get_embeddings(indexes, sin_pattern, weights_norm=None, return_emb_values=False):
    """
    Calculates the ground truth embeddings for given ground truth batch
    :param indexes: ground truth label of instances [BSxWxH]
    :param sin_pattern: tensor with sins patterns [NxWxH]
    :param weights_norm: function for weight normalization
    :param return_emb_values: if True function return embedding vectors for each obj [BSxNxMAXObj]
    :return: return [BSxNxWxH] return ground truth embeddings for each pixel
    """
    device = indexes.device
    nobj = int(torch.max(indexes).item() + 1)
    nsamples, xdim, ydim = indexes.shape

    nemb = sin_pattern.size(0)
    t = sin_pattern.transpose(1, 0).transpose(2, 1).view(1, -1, nemb).repeat(nsamples, 1, 1)

    indexes_raw = indexes.to(device)
    indexes = indexes_raw.long()
    w = torch.zeros(nsamples, nobj).to(device)
    w = w.scatter_add(1, indexes.view(nsamples, -1), torch.ones_like(indexes_raw).view(nsamples, -1))
    e = torch.zeros(nsamples, nobj, nemb).to(device)
    e = e.scatter_add(1, indexes.view(nsamples, -1, 1).repeat(1, 1, nemb), t)
    w[w == 0] = 1.
    e = e / w.unsqueeze(-1)
    # here we get embedding values
    if return_emb_values:
        return e

    if weights_norm is not None:
        w = weights_norm(w)

    w = torch.gather(w, 1, indexes.view(nsamples, -1))
    e = torch.gather(e, 1, indexes.view(nsamples, -1, 1).repeat(1, 1, nemb))
    e = e.transpose(2, 1).contiguous()

    return e.view(nsamples, nemb, xdim, ydim), w.view(nsamples, xdim, ydim)


def flat_pattern_rdm(num1, num2, variance):
    alpha = (torch.rand(num1) - 0.5) * variance
    phase = torch.rand(num1)
    sins1 = [AddSine(a, 0, ph) for a, ph in zip(alpha, phase)]

    alpha = (torch.rand(num2) - 0.5) * variance
    phase = torch.rand(num2)
    sins2 = [AddSine(0, a, ph) for a, ph in zip(alpha, phase)]

    return torch.nn.Sequential(*(sins1 + sins2))


class Embedding(object):
    """
    Embedding class controls the embedding procedure
    """

    def __init__(self, sins, dims, weights_norm=None):
        """
        Create Embedding object
        :param sins: list or torch.nn.Sequential
        :param dims: the size of batch in image
        :param weights_norm: instance weight function
        """
        if isinstance(sins, torch.nn.Sequential):
            self.sins = sins
        elif isinstance(sins, list):
            # Predefined values
            self.sins = torch.nn.Sequential(*[AddSine(a, b, p) for a, b, p in sins])
        else:
            raise RuntimeError("sins are give in unknown type")
        self.sin_pattern = sins(
            torch.ones((1, 1, dims[0], dims[1])))[0, 1:]
        self.weights_norm = weights_norm

    def fit(self, ground_truth_labels, epsilon=0.5, niter=400, lr=100.):
        """
        Fits the best params of sins with respect to a training set X
        :param ground_truth_labels: training set as pytorch dataset [BSxWxH]
        :param lr: learning rate of gradient decent
        :param epsilon: 
        :param niter: number of optimization iterations
        :return: it updates internal states of the Embedder and return sins params as list and errors
        """
        assert isinstance(ground_truth_labels,torch.Tensor), "Expected ground_truth_labels as torch.Tensor"
        assert 3 == len(ground_truth_labels.shape), "ground_truth_labels should be in foramt [BSxWxH]"
        # Re arrange data from 1 to N (important for the loss function)
        numpy_gt = torch.zeros_like(ground_truth_labels)
        for im in range(numpy_gt.shape[0]):
            cc, _ = torch.sort(torch.unique(ground_truth_labels[im]))
            for idx, un in enumerate(cc[1:]):
                numpy_gt[im][ground_truth_labels[im] == int(un.item())] = idx + 1

        indexes_raw = numpy_gt.long()
        number_of_object_per_image = (indexes_raw.max(dim=1)[0].max(dim=1)[0]).long()
        indexes = indexes_raw.float()

        optimizer = torch.optim.SGD(self.sins.parameters(), lr=lr)
        errors = []
        for i in range(niter):
            # zero grad
            optimizer.zero_grad()

            ones = torch.ones((1, 1, indexes.shape[1], indexes.shape[2])).to(indexes_raw.device)
            sin_pat = self.sins(ones)[0, 1:]

            e = get_embeddings(indexes, sin_pat, weights_norm=None, return_emb_values=True)
            emb_loss = 0
            for ex, no in zip(e, number_of_object_per_image):
                pwd = pairwise_distances(ex)
                pwd[no.item():, :] = epsilon * 2.
                pwd[:, no.item():] = epsilon * 2.
                pwd[torch.eye(pwd.shape[0]) == 1] = epsilon * 2.
                res = epsilon - pwd
                res = torch.clamp(res, 0.)
                emb_loss += torch.mean(res)

            errors += [[emb_loss.item()]]
            loss = emb_loss
            loss.backward()
            optimizer.step()

        # update the sins patterns
        self.sin_pattern = self.sins(
            torch.ones((1, 1, self.sin_pattern.shape[1], self.sin_pattern.shape[2])))[0, 1:]

        sins_in_text = []
        for s in self.sins:
            sins_in_text += [[s.alpha.item(), s.beta.item(), s.phase.item()]]
        return sins_in_text, errors

    def validate(self, valid, epsilon=0.5):
        """
        Check the quality of ground truth embedding
        :param valid: indexes to validate state [BSxWxH]
        :param epsilon: the minimal distance between embeddings of two objects within an image
        :return: rate of instances having embeddings closer than epsilon against all instances
        """
        nobj = int(valid.max().item() + 1)

        indexes_raw = valid
        number_of_object_per_image = (indexes_raw.max(dim=1)[0].max(dim=1)[0]).long()
        e = get_embeddings(valid, self.sin_pattern.to(valid.device), return_emb_values=True)
        pwd = torch.zeros((int(nobj), int(nobj)))
        for ex, no in zip(e, number_of_object_per_image):
            dist = pairwise_distances(ex)
            dist[no.item():, :] = epsilon * 2.
            dist[:, no.item():] = epsilon * 2.
            pwd[torch.eye(pwd.shape[0]) == 1] = epsilon * 2.
            dist = dist > epsilon
            pwd += dist.cpu().float()
        less_th = (len(number_of_object_per_image) - pwd) * (1 - torch.eye(pwd.shape[0]))
        return torch.sum(less_th) / sum(number_of_object_per_image).float()

    def __call__(self, ground_truth_labels):
        return get_embeddings(ground_truth_labels, self.sin_pattern.to(ground_truth_labels.device), self.weights_norm)
