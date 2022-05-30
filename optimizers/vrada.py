from dependencies import *

from utils import *
from datasets import *
from objectives import *
import time


def VRADA(score_list, closure, batch_size, D, labels, R=10, init_step_size=1, max_epoch=100,
         x0=None, verbose=True, D_test=None, labels_test=None, seed=-1):
    """
        SVRG with fixed step size for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]


    m = 2*n

    if x0 is None and seed < 0:
        x0 = np.zeros(d)
        x = x0.copy()
        x_tilde = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d) * 10
        x = x0.copy()
        x_tilde = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
        x_tilde = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0

    beta = 1/init_step_size


    a = 1/beta
    A = 1/beta

    loss, full_grad = closure(x_tilde, D, labels)
    num_grad_evals += n

    G_ = a*full_grad

    step_size = a




    score_dict = {"epoch": 1}
    score_dict["n_grad_evals"] = num_grad_evals
    score_dict["n_grad_evals_normalized"] = num_grad_evals / n
    score_dict["train_loss"] = loss
    score_dict["grad_norm"] = np.linalg.norm(full_grad)
    score_dict["train_accuracy"] = accuracy(x_tilde, D, labels)
    score_dict["train_loss_log"] = np.log(loss)
    score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
    # score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
    if D_test is not None:
        test_loss = closure(x_tilde, D_test, labels_test, backwards=False)
        score_dict["test_loss"] = test_loss
        score_dict["test_accuracy"] = accuracy(x_tilde, D_test, labels_test)
        score_dict["test_loss_log"] = np.log(test_loss)
        # score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])
    score_list += [score_dict]
    if verbose:
        output = 'Epoch.: %d, Grad. norm: %.2e' % \
                 (1, np.linalg.norm(full_grad))
        output += ', Func. value: %e' % loss
        output += ', Step size: %e' % step_size
        output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n)
        print(output)

    z = x0 - G_
    if np.linalg.norm(z - x0) > R:
        z = x0 + R*(z-x0)/np.linalg.norm(z-x0)
    x_tilde = z.copy()

    G = np.zeros(G_.shape)


    for k in range(2,max_epoch):
        t_start = time.time()

        a = np.sqrt(m*A/2/beta)

        A_ = A

        A = A_ + a

        print(A, a)

        step_size = a/m
        Z = np.zeros(z.shape)


        loss, full_grad = closure(x_tilde, D, labels)
        num_grad_evals = num_grad_evals + n

        last_full_grad = full_grad
        last_x_tilde = x_tilde

        score_dict = {"epoch": k}
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x_tilde, D, labels)
        score_dict["train_loss_log"] = np.log(loss)
        score_dict["grad_norm_log"] = np.log(score_dict["grad_norm"])
        # score_dict["train_accuracy_log"] = np.log(score_dict["train_accuracy"])
        if D_test is not None:
            test_loss = closure(x_tilde, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x_tilde, D_test, labels_test)
            score_dict["test_loss_log"] = np.log(test_loss)
            # score_dict["test_accuracy_log"] = np.log(score_dict["test_accuracy"])
        score_list += [score_dict]
        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (k, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % step_size
            output += ', Num gradient evaluations/n: %f' % (num_grad_evals / n)
            print(output)

        full_grad_norm = np.linalg.norm(full_grad)
        if full_grad_norm <= 1e-12:
            return score_list
        elif full_grad_norm >= 1e10:
            return score_list
        elif np.isnan(full_grad_norm):
            return score_list

        # Create Minibatches:
        minibatches = make_minibatches(n, m, batch_size)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]

            y = A_/A * x_tilde + a/A * z

            # compute the gradients:
            y_grad = closure(y, Di, labels_i)[1]
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk  = y_grad - x_tilde_grad + full_grad
            num_grad_evals = num_grad_evals + 2 * batch_size

            G = G + a*gk


            z = x0 - G/m - G_
            if np.linalg.norm(z - x0) > R:
                z = x0 + R*(z-x0)/np.linalg.norm(z-x0)

            Z = Z + z
        x_tilde = A_/A * x_tilde + a/A * Z/m
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch
    return score_list
