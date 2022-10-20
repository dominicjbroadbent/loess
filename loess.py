import numpy as np

from bisect import bisect
from scipy.spatial.distance import cdist
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from kernels import rbf_kernel

def make_poly_pred_(x, y, anchor, weights, degree):
    # Transform the data and the regression anchor into the correct basis
    x_ = PolynomialFeatures(degree).fit_transform(x)
    
    if anchor.shape[0] == 1:
        anchor = anchor.reshape(-1, 1)
    else:
        anchor = anchor.reshape(1, -1)
    anchor_ = PolynomialFeatures(degree).fit_transform(anchor)
    # Fit the weighted linear regression model on the data
    model = LinearRegression().fit(x_, y, sample_weight = weights)
    
    # Predict at the regression anchor
    y_hat = model.predict(anchor_)
    
    return y_hat

def interpolate_between_anchors(x, y_hat, anchors):
    y_hat_interpolated = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        if x[i] == anchors.min():
            y_hat_interpolated[i] = y_hat[anchors.argmin()]
            continue
        if x[i] == anchors.max():
            y_hat_interpolated[i] = y_hat[anchors.argmax()]
            continue

        # Figure out which anchor points the x is between 
        after_idx = bisect(anchors, x[i])
        before_idx = after_idx - 1

        # Get the coords of the anchor points
        x0 = anchors[before_idx]
        y0 = y_hat[before_idx]

        x1 = anchors[after_idx]
        y1 = y_hat[after_idx]

        # Gradient of interpolating line
        m = ( y1 - y0 ) / ( x1 - x0 )

        # Intercept of interpolating line
        c = y1 - m * x1

        # Interpolate linearly
        y_hat_interpolated[i] = m * x[i] + c
    
    return y_hat_interpolated

def compute_robust_weights_(y, y_hat):
    residuals = y - y_hat
    s = np.median(np.abs(residuals))
    robust_weights = np.clip(residuals / (6.0 * s), -1, 1)
    return (1 - robust_weights ** 2) ** 2

def fit_loess(x, y, anchors = None, degree = 2, kernel = rbf_kernel, alpha = 1, frac = None, robust_iters = 1):
    assert degree >= 0 and isinstance(degree, int), 'degree must be a non-negative integer'
    
    if frac is not None:
        assert  0 < frac < 1, 'frac must be in (0, 1)'
        
    assert robust_iters > 0 and isinstance(robust_iters, int), 'robust_iters must be a positive integer'
    
    # Ensure arrays are 2-dimensional
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    
    # If we don't supply regression anchors, set every point to be one
    anchor_interpolate_flag = True
    if anchors is None:
        anchors = x
        anchor_interpolate_flag = False
    elif len(anchors.shape) == 1:
        anchors = anchors.reshape(-1, 1)
    
    # Compute euclidean distances between each regression anchor and the rest of the data
    dists = cdist(anchors, x, 'euclidean')   
    
    # Initialise robust weights as ones
    robust_weights = np.ones(y.shape[0])
    
    # Array to hold predictions for the regression anchors
    y_hat = np.zeros(anchors.shape[0])
    if frac is None:
        # Compute the weights using all of the distances 
        weights = kernel( dists / alpha )
        
        for robust_iteration in range(robust_iters):
            # For each regression anchor, ignore (approximately) zero weights and predict
            for i in range(anchors.shape[0]):
                idcs = np.where(robust_weights * weights[i, :] > 1e-10)[0]
                y_hat[i] = make_poly_pred_(x[idcs, :], y[idcs],
                                           anchors[i, :], weights[i, idcs],
                                           degree)
            
            # Interpolate linearly between anchor points if they exist
            # and are not just x
            if anchor_interpolate_flag:
                y_hat = interpolate_between_anchors(x, y_hat, anchors)
                
            # Stop recomputing robust weights after appropriate number of iterations
            if (robust_iteration + 1) != robust_iters:
                robust_weights = compute_robust_weights_(y, y_hat)
                
    else:
        # Compute the fraction to take
        m = int(frac * y.shape[0])

        # Get the indices for the sorted distances
        frac_idcs = np.argsort(dists, axis = 1)[:, :m]
        
        for robust_iteration in range(robust_iters):
            # For each regression anchor, compute the sorted weights and predict
            for i in range(anchors.shape[0]):
                idcs = frac_idcs[i, :]
                weights = robust_weights[idcs] * kernel( dists[i, idcs] / alpha )
                y_hat[i] = make_poly_pred_(x[idcs, :], y[idcs],
                                           anchors[i, :], weights,
                                           degree)
            
            # Interpolate linearly between anchor points if they exist
            # and are not just x
            if anchor_interpolate_flag:
                y_hat = interpolate_between_anchors(x, y_hat, anchors)
                
            # Stop recomputing robust weights after appropriate number of iterations
            if (robust_iteration + 1) != robust_iters:
                robust_weights = compute_robust_weights_(y, y_hat)

    return y_hat