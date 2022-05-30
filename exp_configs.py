from haven import haven_utils as hu
import itertools
import numpy as np

def get_benchmark(benchmark,
                  opt_list,
                  batch_size = 1,
                  runs = [0,1,2,3,4],
                  max_epoch=[50],
                  losses=["logistic_loss", "squared_loss", "squared_hinge_loss"]
                 ):

    if benchmark == "mushrooms":
        return {"dataset":["mushrooms"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./8000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "ijcnn":
        return {"dataset":["ijcnn"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./35000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "a1a":
        return {"dataset":["a1a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./1600,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "a2a":
        return {"dataset":["a2a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./2300,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "w8a":
        return {"dataset":["w8a"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./50000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "covtype":
        return {"dataset":["covtype"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./500000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "phishing":
        return {"dataset":["phishing"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1e-4,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "rcv1":
        return {"dataset":["rcv1"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":1./20000,
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}

    elif benchmark == "synthetic_interpolation":
        return {"dataset":["synthetic"],
                "loss_func": losses,
                "opt": opt_list,
                "regularization_factor":0.,
                "margin":[0.1],
                "false_ratio" : [0, 0.1, 0.2],
                "n_samples": [10000],
                "d": [200],
                "batch_size":batch_size,
                "max_epoch":max_epoch,
                "runs":runs}
    else:
        print("Benchmark unknown")
        return

EXP_GROUPS = {}
MAX_EPOCH = 50
RUNS = [0,1,2,3,4]
benchmarks_list = ["mushrooms", "ijcnn", "a1a", "a2a", "w8a", "rcv1", "covtype", "phishing"]
benchmarks_interpolation_list = ["synthetic_interpolation"]

for benchmark in benchmarks_list + benchmarks_interpolation_list:
    EXP_GROUPS["exp_%s" % benchmark] = []


#=== Setting up main experiments ===
for with_seed in [False, True]:

    for batch_size in [1]:
        MAX_EPOCH = 50
        opt_list = []

        # Baseline optimizers
        for eta in [1e-2,5*1e-2,1e-1, 5*1e-1, 1, 5*1,10, 5*10,100]:
            opt_list += [{'name':'SVRG',
                        'r':1/batch_size,
                        'adaptive_termination':0,
                        'init_step_size':eta,
                        'with_seed': with_seed,
                        'R': 100}]
            opt_list += [{'name':'SARAH',
                        'r':1/batch_size,
                        'init_step_size':eta,
                        'with_seed': with_seed,
                        'R': 100}]
            opt_list += [{'name':'VARAG',
                        'init_step_size':eta,
                        'with_seed': with_seed,
                        'R': 100}]
            opt_list += [{'name':'AdaVRAE_NA',
                         'init_step_size': eta,
                         'R': 100,
                         'with_seed': with_seed}]
            opt_list += [{'name':'AdaVRAG_NA',
                          'init_step_size': eta,
                          'R': 100,
                          'with_seed': with_seed}]

        for benchmark in benchmarks_list:
            EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[MAX_EPOCH], runs=RUNS,
            losses=['huber_loss', 'squared_loss', 'logistic_loss'
            ]))

    for batch_size in [1]:
        MAX_EPOCH = 50
        opt_list = []
        opt_list += [{'name':'AdaSVRG_OneStage',
                     'r':1/batch_size,
                     'init_step_size': 0.01,
                     'R': 100,
                     'with_seed': with_seed}]

        opt_list += [{'name':'AdaVRAE_A',
                     'init_step_size': 0.01,
                     'R': 100,
                     'with_seed': with_seed}]

        opt_list += [{'name':'AdaVRAG_A',
                      'init_step_size': 0.01,
                      'R': 100,
                      'with_seed': with_seed}]


        for benchmark in benchmarks_list:
            EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[MAX_EPOCH], runs=RUNS,
            losses=['huber_loss', 'logistic_loss', 'squared_loss'
            ]))


    for batch_size in [1]:
        MAX_EPOCH = 30
        opt_list = []
        for eta in [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
            opt_list += [{'name':'VRADA',
                         'init_step_size': eta,
                         'with_seed': with_seed,
                         'R': 100
                         }]
        for benchmark in benchmarks_list:
            EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[MAX_EPOCH], runs=RUNS,
            losses=['huber_loss', 'logistic_loss', 'squared_loss'
            ]))

    for batch_size in [1]:
        MAX_EPOCH = 50
        opt_list = []
        for eta in [1e-2,5*1e-2,1e-1, 5*1e-1, 1, 5*1,10, 5*10,100]:
            opt_list += [{'name':'SVRG_PP',
                        'r':1/batch_size,
                        'adaptive_termination':0,
                        'init_step_size':eta,
                        'with_seed': with_seed,
                        'R': 100}]
        opt_list += [{'name':'AdaSVRG_MultiStage',
                     'r':1/batch_size,
                     'init_step_size': 0.01,
                     'with_seed': with_seed,
                     'R': 100}]
        for benchmark in benchmarks_list:
            EXP_GROUPS['exp_%s' % benchmark] += hu.cartesian_exp_group(get_benchmark(benchmark, opt_list, batch_size=batch_size, max_epoch=[MAX_EPOCH], runs=RUNS,
            losses=['huber_loss', 'logistic_loss', 'squared_loss'
            ]))
