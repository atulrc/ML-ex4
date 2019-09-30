# Scientific and vector computation for python
import numpy as np

import utils

def sigGrad(z):
    """
    Computes the gradient of the sigmoid function evaluated at z.
    This should work regardless if z is a matrix or a vector.
    In particular, if z is a vector or matrix, you should return
    the gradient for each element.

    Parameters
    ----------
    z : array_like
        A vector or matrix as input to the sigmoid function.

    Returns
    --------
    g : array_like
        Gradient of the sigmoid function. Has the same shape as z.

    Instructions
    ------------
    Compute the gradient of the sigmoid function evaluated at
    each value of z (z can be a matrix, vector or scalar).

    Note
    ----
    We have provided an implementation of the sigmoid function
    in `utils.py` file accompanying this assignment.
    """

    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = utils.sigmoid(z) * (1 - utils.sigmoid(z))

    # =============================================================
    return g
