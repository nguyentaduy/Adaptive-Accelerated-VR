from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def AdaVRAE_NA(score_list, closure, batch_size, D, labels, R=10,
            init_step_size=1, max_epoch=100, x0=None, verbose=True,
            D_test=None, labels_test=None, seed=-1):
    """
        AdaVRAE (non adaptive) for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]

    A = 1+1/4
    s0 = 0
    m = n
    a = (1/4/n)
    s0 = 0
    c = 3/2



    if x0 is None and seed < 0: # x0 is the starting point
        x0 = np.zeros(d)
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d) * 10
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0
    gamma = 1/init_step_size
    for s in range(1,max_epoch):
        t_start = time.time()
        # set epoch length

        print(A)
        if np.sqrt(a) < 1/2:
            a = np.sqrt(a)
            s0+=1
        else:
            a = (s - s0 - 1 + c)/2/c
            m = n
        print(a)
        A = A - m*a**2

        print("m = ", m)

        if s == 1:
            loss, full_grad = closure(x_tilde, D, labels)
            num_grad_evals = num_grad_evals + n
            print(np.linalg.norm(x_tilde-x0))
            x_bar_grad = full_grad.copy()

        score_dict = {"epoch": s}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = a/gamma
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
            output += ', Step size: %e' % (a/gamma)
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
            x = z - a/gamma*x_bar_grad
            if np.linalg.norm(x - x0) > R:
                x = x0 + R*(x-x0)/np.linalg.norm(x-x0)
            A_ = A + a + a**2
            x_bar = (A*x_bar + a*x + a**2*x_tilde)/A_
            A = A_

            # get the minibatch for this iteration


            if i < int(m)-1:
                indices = minibatches[i]
                Di, labels_i = D[indices, :], labels[indices]
                num_grad_evals = num_grad_evals + 2 * len(indices)
                # compute the gradients:
                loss_temp, x_bar_grad_i = closure(x_bar, Di, labels_i)
                x_tilde_grad_i = closure(x_tilde, Di, labels_i)[1]
                x_bar_grad_ = x_bar_grad_i - x_tilde_grad_i + full_grad
                grad_norm_diff = np.linalg.norm(x_bar_grad_ - x_bar_grad)
                x_bar_grad = x_bar_grad_.copy()
            else:
                loss, full_grad = closure(x_bar, D, labels)
                num_grad_evals = num_grad_evals + n

                x_bar_grad_ = full_grad
                grad_norm_diff = np.linalg.norm(x_bar_grad_ - x_bar_grad)
                x_bar_grad = x_bar_grad_.copy()

            # print(grad_norm_diff)

            gamma_ = gamma
            z = gamma/gamma_*z + (1-gamma/gamma_)*x - a/gamma_*x_bar_grad
            if np.linalg.norm(z - x0) > R:
                z = x0 + R*(z-x0)/np.linalg.norm(z-x0)

            gamma = gamma_

        x = x_bar.copy()
        x_tilde = x_bar.copy()

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list

def AdaVRAE_A(score_list, closure, batch_size, D, labels, R=10,
            init_step_size=1, max_epoch=100, x0=None, verbose=True,
            D_test=None, labels_test=None, seed=-1):
    """
        AdaVRAE (adaptive) for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]

    A = 1+1/4
    s0 = 0
    m = n
    a = (1/4/n)
    s0 = 0
    c = 3/2



    if x0 is None and seed < 0: # x0 is the starting point
        x0 = np.zeros(d)
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d) * 10
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x = x0.copy()
        z = x0.copy()
        x_tilde = x0.copy()
        x_bar = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')

    num_grad_evals = 0
    gamma = init_step_size
    eta = R
    for s in range(1,max_epoch):
        t_start = time.time()
        # set epoch length

        print(A)
        if np.sqrt(a) < 1/2:
            a = np.sqrt(a)
            s0+=1
        else:
            a = (s - s0 - 1 + c)/2/c
            m = n
        print(a)
        A = A - m*a**2


        print("m = ", m)
        if s == 1:
            loss, full_grad = closure(x_tilde, D, labels)
            num_grad_evals = num_grad_evals + n
            print(np.linalg.norm(x_tilde-x0))
            x_bar_grad = full_grad.copy()

        score_dict = {"epoch": s}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = a/gamma
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
            output += ', Step size: %e' % (a/gamma)
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
            x = z - a/gamma*x_bar_grad
            if np.linalg.norm(x - x0) > R:
                x = x0 + R*(x-x0)/np.linalg.norm(x-x0)
            A_ = A + a + a**2
            x_bar = (A*x_bar + a*x + a**2*x_tilde)/A_
            A = A_

            if i < int(m)-1:
                indices = minibatches[i]
                Di, labels_i = D[indices, :], labels[indices]
                num_grad_evals = num_grad_evals + 2 * len(indices)
                # compute the gradients:
                loss_temp, x_bar_grad_i = closure(x_bar, Di, labels_i)
                x_tilde_grad_i = closure(x_tilde, Di, labels_i)[1]
                x_bar_grad_ = x_bar_grad_i - x_tilde_grad_i + full_grad
                grad_norm_diff = np.linalg.norm(x_bar_grad_ - x_bar_grad)
                x_bar_grad = x_bar_grad_.copy()
            else:
                loss, full_grad = closure(x_bar, D, labels)
                num_grad_evals = num_grad_evals + n

                x_bar_grad_ = full_grad
                grad_norm_diff = np.linalg.norm(x_bar_grad_ - x_bar_grad)
                x_bar_grad = x_bar_grad_.copy()
            # print(grad_norm_diff)

            gamma_ = np.sqrt(gamma**2 + a**2*grad_norm_diff**2/eta**2)
            z = gamma/gamma_*z + (1-gamma/gamma_)*x - a/gamma_*x_bar_grad
            if np.linalg.norm(z - x0) > R:
                z = x0 + R*(z-x0)/np.linalg.norm(z-x0)

            gamma = gamma_

        x = x_bar.copy()
        x_tilde = x_bar.copy()

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
