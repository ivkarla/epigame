from numpy import delete

def exclude_node_from_cm(cm_list, channel_id):
    """Excludes a node from connectivity matrices of cm_list epochs.
    Arguments:
        cm_list (list of numpy arrays): list of connectivity matrices saved in the PREP file under X;
                                        First half of the matrices are from seizure epochs, second half form baseline epochs.
        channel_id (int or list): index of the node to be removed
    Returns the updated list of matrices.
    """
    reduced_cms = []
    for cm in cm_list:
      cm = delete(cm, channel_id, 0)
      cm = delete(cm, channel_id, 1)
      reduced_cms.append(cm)
    return reduced_cms
