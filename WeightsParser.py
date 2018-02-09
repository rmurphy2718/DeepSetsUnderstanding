# #######################################################
# Project: Understanding the Deep Sets paper through implementation.
#          Experiment 1: Rotated Gaussians.
# ---------------------------------------------------------------------
#
# WeightsParser.py (NOT MY CODE)
#
# FROM NEURAL FINGERPRINT: NOT MY CODE
#  From Duvenaud, Maclaurin,etc.
#
# This will assist in taking a huge vector of weights
# and returning matrices of the desired shape
#
# Because: grad uses a vector, but score uses mats
# #######################################################


class WeightsParser(object):
    """A kind of dictionary of weights shapes,
       which can pick out named subsets from a long vector.
       Does not actually store any weights itself."""
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.N = 0
    #
    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)
    #
    def get(self, vect, name):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)
    #
    def set(self, vect, name, value):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, _ = self.idxs_and_shapes[name]
        vect[idxs] = np.ravel(value)
    #
    def __len__(self):
        return self.N
