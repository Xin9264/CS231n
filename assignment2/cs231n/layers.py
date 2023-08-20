from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    hidden_layer = x.reshape(x.shape[0], -1)
    out = np.dot(hidden_layer, w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)
    # print(f"dx.shape is {x.shape}")
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
    db = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = x.copy()
    out[out < 0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    probability = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    loss = np.sum(-np.log(probability[range(len(y)), y])) / x.shape[0]
    dx = probability.copy()
    dx[range(len(y)), y] -= 1
    dx = dx / x.shape[0]
    # probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    # probs /= np.sum(probs, axis=1, keepdims=True)
    # N = x.shape[0]
    # loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    # dx = probs.copy()
    # dx[np.arange(N), y] -= 1
    # dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        x_normalized = (x - sample_mean) / (np.sqrt(sample_var + eps))

        # out = (x_normalized - gamma) / beta
        out = gamma * x_normalized + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, bn_param)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        scale = gamma / np.sqrt(running_var + eps)
        # normalized_data = (x - running_mean) / running_var
        out = x * scale + (beta - running_mean * scale)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_normalized, mean, var, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = x.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout*x_normalized, axis=0)
    dx_normalized = dout * gamma
    dvar = np.sum(-0.5 * (x - mean) / np.power(var + eps, 1.5), axis=0)
    dmean = np.sum(dx_normalized * (-1)/np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
    dx = dx_normalized * (1 / np.sqrt(var + eps)) + dvar * (2 * (x - mean) / N) + dmean / N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, x_normalized, mean, var, gamma, beta, bn_param = cache
    # D, N = x.shape
    N = dout.shape[0]
    eps = bn_param.get('eps', 1e-5)
    dgamma = np.sum(dout * x_normalized, axis=0)
    dbeta = np.sum(dout, axis=0)
    S = lambda x: np.sum(x, axis=0)
    # dx_hat = dout * gamma
    dx = (1.0 / N) * gamma * (var + eps)**(-1.0 / 2.0) * (N * dout - S(dout) - (x - mean) * (var + eps)**(-1.0) * S(dout * (x - mean)))
    # gamma, x, sample_mean, sample_var, eps, x_hat = cache
    # sample_mean = mean
    # sample_var = var
    # x_hat = x_normalized
    # N = x.shape[0]
    # dx_hat = dout * gamma
    # dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * np.power(sample_var + eps, -1.5), axis=0)
    # dmean = np.sum(dx_hat * -1 / np.sqrt(sample_var + eps), axis=0) + dvar * np.mean(-2 * (x - sample_mean), axis=0)
    # dx = 1 / np.sqrt(sample_var + eps) * dx_hat + dvar * 2.0 / N * (x - sample_mean) + 1.0 / N * dmean
    # dgamma = np.sum(x_hat * dout, axis=0)
    # dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    x_mu = x - sample_mean
    var_inv = 1 / (np.sqrt(sample_var + eps))
    x_norm = x_mu * var_inv

    out = gamma * x_norm + beta
    cache = (x, x_norm, x_mu, var_inv, gamma, beta)
    # sample_mean = np.mean(x, axis=1, keepdims=True)
    # sample_var = np.var(x, axis=1, keepdims=True)
    #
    # x_normalized = (x - sample_mean) / (np.sqrt(sample_var + eps))
    #
    # # out = (x_normalized - gamma) / beta
    # out = gamma * x_normalized + beta
    # cache = (x, x_normalized, sample_mean, sample_var, gamma, beta, ln_param)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, x_norm, x_mu, var_inv, gamma, beta = cache
    N, D = dout.shape
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    dxnorm = dout * gamma
    dxmu = dxnorm * var_inv
    dvar_inv = np.sum(dxnorm * x_mu, axis=1, keepdims=True)

    dvar = dvar_inv * -0.5 * var_inv **3
    dx = dxmu

    dxmu += dvar * 2 / D * x_mu
    dmu = -1 * np.sum(dxmu, axis=1, keepdims=True)

    dx += 1/D * dmu
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = np.random.rand(*x.shape) < p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    stride = conv_param['stride']
    pad_num = conv_param['pad']
    H_prime = 1 + (H + 2 * pad_num - HH) // stride 
    W_prime = 1 + (W + 2 * pad_num - WW) // stride 
    x_pad = np.pad(x, ((0, 0),(0, 0), (pad_num, pad_num), (pad_num, pad_num)))
    out = np.zeros((N, F, H_prime, W_prime))
    for i in range(N):
        for j in range(F):
          for h in range(H_prime):
              for w_ in range(W_prime):
                  h1 = h * stride
                  h2 = h * stride + HH
                  w1 = w_ * stride
                  w2 = w1 + WW
                  x_slice = x_pad[i, :, h1:h2, w1:w2]
                  out[i, j, h, w_] = np.sum(x_slice * w[j, :, :, :]
                                             ) + b[j]          
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, w, b, conv_param = cache
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    (_, _, H_out, W_out) = dout.shape
    stride = conv_param['stride']
    padding = conv_param["pad"]
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), ((padding, padding))))
    dx_pad = np.pad(dx, ((0, 0), (0, 0), (padding, padding), ((padding, padding))))
    for n in range(N):
        for f in range(F):
            for hout in range(H_out):
                for wout in range(W_out):
                  h1 = hout * stride
                  h2 = h1 + HH
                  w1 = wout * stride
                  w2 = w1 + WW
                  x_slice = x_padded[n, :, h1:h2, w1:w2]
                  dw[f, :, :, :] += x_slice * dout[n, f, hout, wout] # x_slice为当前梯度，再乘以后面传过来的梯度以此传播
                  db[f] += 1 * dout[n, f, hout, wout]     
                  dx_pad[n, :, h1:h2, w1:w2] += w[f, :, :, :] * dout[n, f, hout, wout]
    dx = dx_pad[:, :, padding:-padding, padding:-padding]
    # ## compute dx
    # dout_pad = np.pad(dout, ((0, 0), (0, 0), (padding, padding), ((padding, padding))))
   
    # for n in range(N):
    #     for c in range(C):
    #         for hx in range(H):
    #             for wx in range(W):
    #                 for f in range(F):
    #                     for i in range(H_out):
    #                         for j in range(W_out):
    #                           h1 = i * stride
    #                           h2 = h1 + HH
    #                           w1 = j * stride
    #                           w2 = w1 + WW
    #                           # if h1 <= hx < h2 and w1 <= wx < w2:
    #                           dx[n, c, hx, wx] += np.sum(w[f, c, :, :] * dout_pad[n, f, h1:h2, w1:w2])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    H_prime = 1 + (H - pool_height) // stride
    W_prime = 1 + (W - pool_width) // stride
    out = np.zeros((N, C, H_prime, W_prime))
    for n in range(N):
        for c in range(C):
          for i in range(H_prime):
              for j in range(W_prime):
                  h1 = i * stride
                  h2 = h1 + pool_height
                  w1 = j * stride
                  w2 = w1 + pool_width
                  out[n, c, i, j] = np.max(x[n, c, h1:h2, w1:w2])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache
    N, C, H, W = x.shape
    _, _, H_prime, W_prime = dout.shape
    stride = pool_param['stride']
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(H_prime):
                for j in range(W_prime):
                    h1 = i * stride
                    h2 = h1 + pool_height
                    w1 = j * stride
                    w2 = w1 + pool_width
                    x_slice = x[n, c, h1:h2, w1:w2]
                    max_x = np.max(x_slice)
                    dmax_x = np.where(x_slice == max_x, 1, 0)
                    dx[n, c, h1:h2, w1:w2] += dmax_x * dout[n, c, i, j]
            

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # out = np.zeros_like(x)
    N, C, H, W = x.shape
    x = np.moveaxis(x, 1, -1).reshape(-1, C)
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = np.moveaxis(out.reshape(N, H, W, C), -1, 1)    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout = np.moveaxis(dout, 1, -1).reshape(-1, C)
    dout, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = np.moveaxis(dout.reshape(N, H, W, C), -1, 1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


# def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
#     """Computes the forward pass for spatial group normalization.

#     In contrast to layer normalization, group normalization splits each entry in the data into G
#     contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
#     are then applied to the data, in a manner identical to that of batch normalization and layer
#     normalization.

#     Inputs:
#     - x: Input data of shape (N, C, H, W)
#     - gamma: Scale parameter, of shape (1, C, 1, 1)
#     - beta: Shift parameter, of shape (1, C, 1, 1)
#     - G: Integer mumber of groups to split into, should be a divisor of C
#     - gn_param: Dictionary with the following keys:
#       - eps: Constant for numeric stability

#     Returns a tuple of:
#     - out: Output data, of shape (N, C, H, W)
#     - cache: Values needed for the backward pass
#     """
#     out, cache = None, None
#     eps = gn_param.get("eps", 1e-5)
#     ###########################################################################
#     # TODO: Implement the forward pass for spatial group normalization.       #
#     # This will be extremely similar to the layer norm implementation.        #
#     # In particular, think about how you could transform the matrix so that   #
#     # the bulk of the code is similar to both train-time batch normalization  #
#     # and layer normalization!                                                #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     N, C, H, W = x.shape
#     x = x.reshape(N*G, -1)
#     gamma = np.tile(gamma, (N, 1, H, W)).reshape(N*G, -1)
#     beta = np.tile(beta, (N, 1, H, W)).reshape(N*G, -1)
#     out, cache = layernorm_forward(x, gamma, beta, gn_param)
#     out = out.reshape(N, C, H, W)
#     cache = (G, cache)
#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return out, cache


# def spatial_groupnorm_backward(dout, cache):
#     """Computes the backward pass for spatial group normalization.

#     Inputs:
#     - dout: Upstream derivatives, of shape (N, C, H, W)
#     - cache: Values from the forward pass

#     Returns a tuple of:
#     - dx: Gradient with respect to inputs, of shape (N, C, H, W)
#     - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
#     - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
#     """
#     dx, dgamma, dbeta = None, None, None

#     ###########################################################################
#     # TODO: Implement the backward pass for spatial group normalization.      #
#     # This will be extremely similar to the layer norm implementation.        #
#     ###########################################################################
#     # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#     G, layer_cache = cache
#     N, C, H, W = dout.shape
#     dout = dout.reshape(N * G, -1)
#     dx, dgamma, dbeta = layernorm_backward(dout, layer_cache)
#     dgamma = dgamma[None, C, None, None]
#     dbeta = dbeta[None, C, None, None]
#     dx = dx.reshape(N, C, H, W)

#     # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#     ###########################################################################
#     #                             END OF YOUR CODE                            #
#     ###########################################################################
#     return dx, dgamma, dbeta

def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner 
    identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_g = np.reshape(x, (x.shape[0]*G, -1))  # N*G,H*W*(C/G)
    mu = np.mean(x_g, axis=1, keepdims=True) # N*G,1
    var = np.var(x_g, axis=1, keepdims=True) # N*G,1
    var_inv = 1 / np.sqrt(var + eps) # N*G,1
    x_mu = x_g - mu # N*G,H*W*(C/G)
    x_norm = x_mu * var_inv # N*G,H*W*(C/G)
    x_norm = np.reshape(x_norm, x.shape)
    out = gamma * x_norm + beta
    cache = (gamma, x_norm, x_mu, var_inv) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    gamma, x_norm, x_mu, var_inv = cache
    
    dgamma = np.sum(dout * x_norm, axis=(0,2,3))[None, ..., None, None] # (N, C, H, W) -> C,
    dxnorm = dout * gamma # (N, C, H, W)
    dbeta = np.sum(dout, axis=(0,2,3))[None, ..., None, None] # (N, C, H, W) -> C,

    dxnorm = np.reshape(dxnorm, x_mu.shape) # N*G,H*W*(C/G)
    dxmu = dxnorm * var_inv  # N*G,H*W*(C/G)
    dvar_inv = np.sum(dxnorm * x_mu, axis=1, keepdims=True)  # N*G,H*W*(C/G) -> N*G,

    dvar = dvar_inv * -0.5 * var_inv ** 3 # N*G,
    dx = dxmu

    dxmu += dvar * 2/(x_mu.shape[1]) * x_mu
    dmu = -1 * np.sum(dxmu, axis=1, keepdims=True)
    
    dx += 1/(x_mu.shape[1]) * dmu 
    dx = np.reshape(dx, dout.shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta