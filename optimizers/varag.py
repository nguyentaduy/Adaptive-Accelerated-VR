from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def VARAG(score_list, closure, batch_size, D, labels, R=10,
            init_step_size=1, max_epoch=100, x0=None, verbose=True,
            D_test=None, labels_test=None, seed=-1):
    """
        AdaSVRG for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]

    s0 = int(np.floor(np.log2(n)) + 1)

    ps = 1/2

    if x0 is None and seed < 0: # x0 is the starting point
        x0 = np.zeros(d)
        x = x0.copy()
        x_tilde = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d)  * 10
        x = x0.copy()
        x_tilde = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
        x_tilde = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0

    for s in range(1,max_epoch):
        t_start = time.time()
        # set epoch length
        if s <= s0:
            as_ = 1/2
            m = 2**(s-1)
        else:
            as_ = 2/(s - s0+4)
            m = 2**(s0-1)
        step_size = init_step_size/(2*as_)

        u = np.zeros(d)
        theta = 0

        loss, full_grad = closure(x_tilde, D, labels)
        num_grad_evals = num_grad_evals + n

        print("m = ", m)
        xt = x.copy()
        x_up = x_tilde.copy()

        score_dict = {"epoch": s}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = step_size
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x_tilde, D, labels)

        if D_test is not None:
            test_loss = closure(x_tilde, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x_tilde, D_test, labels_test)

        score_list += [score_dict]
        if verbose:
            output = 'Epoch.: %d, Grad. norm: %.2e' % \
                     (s, np.linalg.norm(full_grad))
            output += ', Func. value: %e' % loss
            output += ', Step size: %e' % (step_size)
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
        for i in range(int(m)):
            x_low = (1-as_-ps)*x_up + as_*xt + ps*x_tilde

            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * len(indices)

            # compute the gradients:
            loss_temp, x_low_grad_i = closure(x_low, Di, labels_i)
            x_tilde_grad_i = closure(x_tilde, Di, labels_i)[1]

            gt = x_low_grad_i - x_tilde_grad_i + full_grad

            xt_ = xt - step_size * gt
            if np.linalg.norm(xt_ - x0) > R:
                xt_ = x0 + R*(xt_-x0)/np.linalg.norm(xt_-x0)
            x_up = (1-as_-ps)*x_up + as_*xt_ + ps*x_tilde

            xt = xt_.copy()

            if i < m-1:
                u += (as_+ps)*x_up
                theta += (as_+ps)
            else:
                u += x_up
                theta += 1

        x = xt.copy()
        x_tilde = u/theta

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
