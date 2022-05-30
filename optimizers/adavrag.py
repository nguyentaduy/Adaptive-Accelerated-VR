from dependencies import *
from utils import *
from datasets import *
from objectives import *
import time




def AdaVRAG_NA(score_list, closure, batch_size, D, labels, R=10,
            init_step_size=1, max_epoch=100, x0=None, verbose=True,
            D_test=None, labels_test=None, seed=-1):
    """
        AdaVRAG (non adaptive) for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]



    as_ = 1-1/(4*n)
    m = n
    s0 = np.ceil(np.log2(np.log2(4*n)))
    a0 = 1 - (4*n)**(-0.5**s0)
    p0 = 3/8

    u_ = (3+np.sqrt(33))/4

    if x0 is None and seed < 0: # x0 is the starting point
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
    Dt = 1/init_step_size
    for s in range(1,max_epoch):
        t_start = time.time()
        # set epoch length
        if s <= s0:
            as_ = 1 - (4*n)**(-0.5**s)
            qs = 1/(1-as_)/as_/2
        else:
            as_ = u_/(s - s0+2*u_)/2
            qs = (2-as_)*as_/(1-as_)/p0/2

        ps = (2-as_)*as_/(1-as_)/qs # ps in

        Dt = 1/init_step_size*ps

        zs = 1 - as_


        u = np.zeros(d)
        theta = 0

        loss, full_grad = closure(x_tilde, D, labels)
        num_grad_evals = num_grad_evals + n

        print(np.linalg.norm(x_tilde))


        print("m = ", m)
        xt = x.copy()
        x_up = x_tilde.copy()

        score_dict = {"epoch": s}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = 1/(qs*Dt)
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
            output += ', Step size: %e' % (1/(qs*Dt))
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
            x_low = (1-as_-zs)*x_up + as_*xt + zs*x_tilde

            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * len(indices)

            # compute the gradients:
            loss_temp, x_low_grad_i = closure(x_low, Di, labels_i)
            x_tilde_grad_i = closure(x_tilde, Di, labels_i)[1]

            gt = x_low_grad_i - x_tilde_grad_i + full_grad

            xt_ = xt - 1/(qs*Dt) * gt

            if np.linalg.norm(xt_ - x0) > R:
                xt_ = x0 + R*(xt_-x0)/np.linalg.norm(xt_-x0)


            x_up = (1-as_-zs)*x_up + as_*xt_ + zs*x_tilde
            xt = xt_.copy()

            if i < m-1:
                u += (as_+zs)*x_up
                theta += (as_+zs)
            else:
                u += x_up
                theta += 1

        x = xt.copy()
        x_tilde = u/theta

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list


def AdaVRAG_A(score_list, closure, batch_size, D, labels, R=10,
            init_step_size=1, max_epoch=100, x0=None, verbose=True,
            D_test=None, labels_test=None, seed=-1, option=2):
    """
        AdaVRAG (adaptive) for solving finite-sum problems
        Closure: a PyTorch-style closure returning the objective value and it's gradient.
        batch_size: the size of minibatches to use.
        D: the set of input vectors (usually X).
        labels: the labels corresponding to the inputs D.
        init_step_size: step-size to use
        n, d: size of the problem
    """
    n = D.shape[0]
    d = D.shape[1]



    as_ = 1-1/(4*n)
    m = n
    s0 = np.ceil(np.log2(np.log2(4*n)))
    a0 = 1 - (4*n)**(-0.5**s0)
    p0 = 3/8

    u_ = (3+np.sqrt(33))/4


    if x0 is None and seed < 0: # x0 is the starting point
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
    Dt = init_step_size

    eta = R
    for s in range(1,max_epoch):
        t_start = time.time()
        # set epoch length
        if s <= s0:
            as_ = 1 - (4*n)**(-0.5**s)
            qs = 1/(1-as_)/as_/2
        else:
            as_ = u_/(s - s0+2*u_)
            qs = (2-as_)*as_/(1-as_)/p0/2

        ps = (2-as_)*as_/(1-as_)/qs # ps in
        zs = 1 - as_


        u = np.zeros(d)
        theta = 0

        loss, full_grad = closure(x_tilde, D, labels)
        num_grad_evals = num_grad_evals + n

        print(np.linalg.norm(x_tilde))


        print("m = ", m)
        xt = x.copy()
        x_up = x_tilde.copy()

        score_dict = {"epoch": s}
        score_dict["optimizer"] = 1
        score_dict["m"] = m
        score_dict["n_grad_evals"] = num_grad_evals
        score_dict["step_size"] = 1/(qs*Dt)
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
            output += ', Step size: %e' % (1/(qs*Dt))
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
            x_low = (1-as_-zs)*x_up + as_*xt + zs*x_tilde

            # get the minibatch for this iteration
            indices = minibatches[i]
            Di, labels_i = D[indices, :], labels[indices]
            num_grad_evals = num_grad_evals + 2 * len(indices)

            # compute the gradients:
            loss_temp, x_low_grad_i = closure(x_low, Di, labels_i)
            x_tilde_grad_i = closure(x_tilde, Di, labels_i)[1]

            gt = x_low_grad_i - x_tilde_grad_i + full_grad

            xt_ = xt - 1/(qs*Dt) * gt

            if np.linalg.norm(xt_ - x0) > R:
                xt_ = x0 + R*(xt_-x0)/np.linalg.norm(xt_-x0)


            x_up = (1-as_-zs)*x_up + as_*xt_ + zs*x_tilde

            if option==1:
                Dt = Dt +np.linalg.norm(xt_-xt)**2/eta**2
            elif option==2:
                Dt = Dt*np.sqrt(1+np.linalg.norm(xt_-xt)**2/eta**2)
            xt = xt_.copy()

            if i < m-1:
                u += (as_+zs)*x_up
                theta += (as_+zs)
            else:
                u += x_up
                theta += 1

        x = xt.copy()
        x_tilde = u/theta

        t_end = time.time()
        time_epoch = t_end - t_start
        score_list[len(score_list) - 1]["time"] = time_epoch

    return score_list
