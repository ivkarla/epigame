from numpy import delete

def exclude_from_cm(cm, channel_id):
    """Excludes a node from a connectivity matrix.
    Arguments:
        cm (numpy array): 2D matrix with connectivity measures between all node pairs
        channel_id (int or list): index of the node to be removed
    Returns the updated matrix.
    """
    cm = delete(cm, channel_id, 0)
    cm = delete(cm, channel_id, 1)
    return cm

def exclude_from_sz_cm(cm_list, channel_id):
    """Excludes a node from connectivity matrices of seizure epochs.
    Arguments:
        cm_list (list of numpy arrays): list of connectivity matrices saved in the PREP file under X;
                                        First half of the matrices are from seizure epochs, second half form basleine epochs.
        channel_id (int or list): index of the node to be removed
    Returns the updated list of matrices.
    """
    result = []
    for i,cm in enumerate(cm_list):
      if i<(len(cm_list)/2):
        # Exclude the node only from seizure matrices, which are set as the first half in the list
        cm = exclude_from_cm(cm, channel_id)
        result.append(cm)
      else:
        result.append(cm)
    return result
