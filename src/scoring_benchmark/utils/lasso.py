import os
import lasso










#Not done yet
def compute_lasso(data, target, alpha, max_iter=1000, tol=1e-4):
    """
    Compute Lasso regression with given data and target.

    Parameters
    ----------
    data : array-like
        Data to be used for Lasso regression.
    target : array-like
        Target to be used for Lasso regression.
    alpha : float
        Regularization strength.
    max_iter : int, optional
        Maximum number of iterations. The default is 1000.
    tol : float, optional
        Tolerance for stopping criteria. The default is 1e-4.

    Returns
    -------
    lasso : Lasso
        Lasso regression model.

    """
    lasso = lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    lasso.fit(data, target)
    return lasso