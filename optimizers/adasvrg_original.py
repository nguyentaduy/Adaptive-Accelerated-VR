from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time

def AdaSVRG_OneStage(score_list, closure, batch_size, D, labels,
            init_step_size=None, max_epoch=100, r=0, x0=None, verbose=True,
            R=10,
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


    if r <= 0:
        m = n
        if verbose:
            print('Info: set m=n by default')
    else:
        m = int(r * n)
        print('Info: set m = ', r, ' n')

    if x0 is None and seed < 0:
        x0 = np.zeros(d)
        x_tilde = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d) * 10
        x_tilde = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x_tilde = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')


    num_grad_evals = 0
    Gk2 = 0

    eta = 2*R/np.sqrt(2)

    for k in range(max_epoch):
        t_start = time.time()

        loss, full_grad = closure(x_tilde, D, labels)
        x = x_tilde.copy()
        num_grad_evals = num_grad_evals + n

        step_size = 1/np.sqrt(Gk2)*eta

        print(np.linalg.norm(x_tilde))


        # initialize running sum of gradient norms
        Gk2 = 0

        print("m = ", m)


        score_dict = {"epoch": k}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = step_size
        score_dict["n_grad_evals_normalized"] = num_grad_evals / n
        score_dict["train_loss"] = loss
        score_dict["grad_norm"] = np.linalg.norm(full_grad)
        score_dict["train_accuracy"] = accuracy(x, D, labels)
        if D_test is not None:
            test_loss = closure(x, D_test, labels_test, backwards=False)
            score_dict["test_loss"] = test_loss
            score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)

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

        u = np.zeros(d)
        for i in range(m):
            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * len(indices)

            # compute the gradients:
            loss_temp, x_grad = closure(x, Di, labels_i)
            x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
            gk = x_grad - x_tilde_grad + full_grad
            Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)

            step_size = 1/np.sqrt(Gk2)*eta

            x -= step_size * gk

            if np.linalg.norm(x - x0) > R:
                x = x0 + R*(x-x0)/np.linalg.norm(x-x0)
            u += x
        x_tilde = u/m
        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list

def AdaSVRG_MultiStage(score_list, closure, batch_size, D, labels,
            init_step_size=None, max_epoch=100, K=3, x0=None, verbose=True,
            R=10,
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



    if x0 is None and seed < 0:
        x0 = np.zeros(d)
        x_tilde = x0.copy()
    elif x0 is None:
        np.random.seed(seed)
        x0 = np.random.rand(d) * 10
        x_tilde = x0.copy()
    elif isinstance(x0, np.ndarray) and x0.shape == (d,):
        x_tilde = x0.copy()
    else:
        raise ValueError('x0 must be a numpy array of size (d, )')


    num_grad_evals = 0
    eta = 2*R/np.sqrt(2)

    total = 0
    for e in range(1,max_epoch):
        m = 2**(e+1)
        b = False

        for k in range(K):
            t_start = time.time()

            loss, full_grad = closure(x_tilde, D, labels)
            x = x_tilde.copy()
            num_grad_evals = num_grad_evals + n

            # initialize running sum of gradient norms
            Gk2 = 0

            print("m = ", m)

            step_size = 1/np.sqrt(Gk2)*eta

            score_dict = {"epoch": k}
            score_dict["optimizer"] = 1
            score_dict["m"] = m
            score_dict["n_grad_evals"] = num_grad_evals
            score_dict["step_size"] = step_size
            score_dict["n_grad_evals_normalized"] = num_grad_evals / n
            score_dict["train_loss"] = loss
            score_dict["grad_norm"] = np.linalg.norm(full_grad)
            score_dict["train_accuracy"] = accuracy(x, D, labels)
            if D_test is not None:
                test_loss = closure(x, D_test, labels_test, backwards=False)
                score_dict["test_loss"] = test_loss
                score_dict["test_accuracy"] = accuracy(x, D_test, labels_test)

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

            u = np.zeros(d)

            if total > max_epoch*n:
                b = True
                break

            for i in range(m):
                # get the minibatch for this iteration
                indices = minibatches[i]
                Di, labels_i = D[indices, :], labels[indices]
                num_grad_evals = num_grad_evals + 2 * len(indices)

                # compute the gradients:
                loss_temp, x_grad = closure(x, Di, labels_i)
                x_tilde_grad = closure(x_tilde, Di, labels_i)[1]
                gk = x_grad - x_tilde_grad + full_grad
                Gk2 = Gk2 + (np.linalg.norm(gk) ** 2)

                step_size = 1/np.sqrt(Gk2)*eta

                x -= step_size * gk

                if np.linalg.norm(x - x0) > R:
                    x = x0 + R*(x-x0)/np.linalg.norm(x-x0)
                u += x
            x_tilde = u/m
            t_end = time.time()
            time_epoch = t_end - t_start
            score_list[len(score_list) - 1]["time"] = time_epoch
            total += m
        if b:
            break

    return score_list
