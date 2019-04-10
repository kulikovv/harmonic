import numpy as np
from skimage.measure import regionprops
from sklearn.cluster import MeanShift


def cluster(embedding_result, bandwidth=0.3, semantic=None, minimal_area=20):
    """
    Cluster embeddings to separate instances
    :param embedding_result: embedding as numpy array
    :param bandwidth: the bandwidth should be equal to epsilon from fit method of embedder
    :param semantic: semantic segmentation mask (in that case each value of mask will be processed separatly)
    :param minimal_area: minimal area of instance
    :return: 
    """
    assert isinstance(embedding_result, np.ndarray), "res should be numpy ndarray"
    assert 3 == len(embedding_result.shape), "res should be [NxWxH] where N is the number of guide functions"

    x = embedding_result.reshape(embedding_result.shape[0], -1).transpose(1, 0)
    l = np.zeros(x.shape[0])

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    if semantic is not None:
        inc = 1
        sem = semantic.reshape(-1)
        assert sem.shape[0] == x.shape[0], "semantic segmentation mask must be of the same size"
    else:
        inc = 0
        sem = np.ones(x.shape[0]).reshape(-1)

    for c in np.unique(sem):
        if 0 == c:
            # ignore background
            continue

        ms.fit(x[sem == c])
        l[sem == c] = ms.labels_ + np.max(l) + inc

    regions = regionprops(l.reshape(embedding_result[0].shape).astype("int32"))
    for r in regions:
        # filter small objects
        if r.area < minimal_area:
            l[l == r.label] = 0

    return l.reshape(embedding_result[0].shape)
