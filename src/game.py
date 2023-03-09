# pylint: disable=no-self-argument, no-member

from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np

from src.data_legacy import record

def classify_epochs(set, node_group, kratio=.1, random_state=31, **mopts):
    """Quantifies connectivity change of specified node group.
    Classifies time frame epochs using the connectivity measures as features. 
    K-fold cross validation scores measure the connectivity change.

    Args:
        set (core.rec): Preprocessed data - connectivity matrices per epoch.
        nodes (set): Node group for which connectivity change is quantified. 
        kratio (float): Ratio of epochs considered in a fold. Defaults to 0.1, which is ~10s of data if epoch span is 1s.
        random_state (int): KFold argument; Controls randomness of each fold. Defaults to 31.

    Returns:
        ndarray of float: Array of scores for each fold.
    """
    X, Y = np.array([np.array((record(x).include(*node_group).T).include(*node_group)).flatten() for x in set.X]), set.y
    model, scaler, k = SVC, StandardScaler, int(round(len(Y)*kratio))
    if random_state == None: random_state = np.random.randint(0xFFFFFFFF)
    C = Pipeline([('scaler', scaler()), ('model', model(**mopts))])
    cv = KFold(k, shuffle=True, random_state=random_state)
    cvs = cross_val_score(C, X, Y, cv=cv)
    return cvs

def evaluation_function(results):
    """Heuristic evaluation function.

    Args:
        results (ndarray): Cross validation score array.

    Returns:
        float: evaluation of results.
    """
    return max(results)*(min(results)/np.average(results)) 

def evaluate_nodes(nodes, labels, results, symbol='<->', f=evaluation_function):
    """Service function. 
    Creates sets with node indices, labels, cross validation scores and a reference function of the scores.

    Args:
        nodes ([type]): [description]
        labels ([type]): [description]
        results (ndarray): Cross validation score array.
        symbol (str): Symbol used to connect node labels within a network. Defaults to '<->'.
        f (function): Function of cross validation scores used to rank the networks based on their connectivity change. Defaults to evaluation_function.

    Returns:
        set: A set of all the arguments.
    """
    tag = symbol.join([labels[n] for n in nodes])
    return (nodes, tag, results, f(results))

def check_until(net, set=1, fall=0, at=-1, limit=None):
    """This function takes a list of analyzed node combinations sorted by their evaluation score.
    The function defines an index of the list until which the node combinations are considered epileptogenic.

    Node combination are iterated and the average evaluation score for each combination is computed. 
    The score is used as a threshold, above which epileptogenic nodes are selected.
    When the score decreases the iteration stops.

    Args:
        net (list): A list of sets returned by evaluate_nodes().
        set (int): Defaults to 1.
        fall (int): Define score threshold (should be the highest score). Defaults to 0.
        at (int): Denotes the index of the set, containing the results for a node group. Defaults to -1, as this is the current saving format.
        limit (int): Define limit until which the nodes are checked. This could be useful for checking node pairs (if many node pairs have the highest score, the WOI could be uninformative).

    Returns:
        int: Index of net list.
    """
    print("Searching for the index of the last best network...")
    l = len(net)
    if limit: l = limit
    while set < l:
        print("Net =", net[:set])
        print("Score =", [n[at] for n in net[:set]][-1])
        score = [n[at] for n in net[:set]][-1]
        if score>=fall:
            fall=score
            set+=1
        else: break
    print(f"Last best network at index {set-2}")
    return set-1
