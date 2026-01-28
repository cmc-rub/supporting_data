import cma
import chargefitter
import numpy as np
from mpi4py import MPI


# get MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# set up the mpi-parallel training problem
# NOTE: parallelization is realized by distributing target systems to the processes and computing local fitnesses, not by parallelizing CMAES
p = chargefitter.generate_multiproblem_mpi("trainset",
                                           fixes = ["EN h1_c1"],                    # for atypes
                                           #fixes = ["EN h1"],                      # for ctypes
                                           #fixes = ["EN h"],                       # for etypes
                                           modelargs = ["topoqeq", "slater", -1, 0.0],
                                           modelkwargs = dict(rule=2),              # for atypes
                                           #modelkwargs = dict(rule=1),             # for ctypes
                                           #modelkwargs = dict(rule=0),             # for etypes
                                           problemkwargs = dict(fit_sigma=True, fit_EN=True, fit_Jii=True, use_q_ref=True,
                                                                param_scalers_qeq = {"sigma": 1e-3,  "Jii": 1.0, "EN": 1.0, "q_base": 1e-3},
                                                               )
                                          )
# set up the mpi-parallel testing/validation problem
p_test = chargefitter.generate_multiproblem_mpi("testset",
                                                fixes = ["EN h1_c1"],
                                                modelargs = ["topoqeq", "slater", -1, 0.0],
                                                modelkwargs = dict(rule=2),
                                                problemkwargs = dict(fit_sigma=True, fit_EN=True, fit_Jii=True, use_q_ref=True,
                                                                    param_scalers_qeq = {"sigma": 1e-3,  "Jii": 1.0, "EN": 1.0, "q_base": 1e-3},
                                                                   )
                                               )
# prepare logfiles, prepare bounds and initial guess for CMAES optimizer
if rank == 0:
    with open("fit_keys.out", "w") as keyfile:
        keyfile.write(' '.join(p.fit_keys))
    with open("fitness.log", "w") as fitfile:
        fitfile.write("FIT_TRAIN"+" "+"FIT_TEST\n")
    test_par_ids = [p.fit_keys.index(k) for k in p_test.fit_keys]
    bounds = [(p.n_params+1)//3*[10.0]+(p.n_params+1)//3*[0.0]+(p.n_params-2)//3*[-2000.0],p.n_params*[2000.0]]
    k_dict = {"sigma":0, "Jii":1, "EN":2}
    x0_dict = {"h" :[[350, 400], [0, 100], [-100, 200]],
               "c" :[[450, 550], [0, 100], [  50, 250]],
               "o" :[[400, 450], [0, 100], [ 450, 650]],
               "n" :[[450, 500], [0, 100], [ 200, 600]],
               "s" :[[500, 550], [0, 100], [ 350, 550]],
               "p" :[[550, 600], [0, 100], [ 100, 500]],
               "f" :[[350, 450], [0, 100], [ 350, 650]],
               "cl":[[400, 500], [0, 100], [ 300, 600]],
               "br":[[450, 550], [0, 100], [ 250, 550]],
               "i" :[[500, 600], [0, 100], [   0, 500]],
               "cu":[[600, 700], [0, 100], [-600,   0]],
               "zn":[[600, 700], [0, 100], [-600,   0]],
               "zr":[[600, 700], [0, 100], [-600,   0]],
               "ag":[[600, 700], [0, 100], [-600,   0]],
               "al":[[600, 700], [0, 100], [-600,   0]],
               "fe":[[600, 700], [0, 100], [-600,   0]],
               "mn":[[600, 700], [0, 100], [-600,   0]],
               "co":[[600, 700], [0, 100], [-600,   0]],
               "ni":[[600, 700], [0, 100], [-600,   0]],
               "cd":[[600, 700], [0, 100], [-600,   0]],
               "xx":[[300, 700], [0, 100], [-600, 600]]
               }
    x0 = []
    for k in p.fit_keys:
        ptype, atype = k.split()
        elem = ''.join([i for i in atype if i.isdigit()==False])
        if elem not in x0_dict.keys():
            guess_bounds = x0_dict["xx"][k_dict[ptype]]
        else:
            guess_bounds = x0_dict[elem][k_dict[ptype]]
        x0.append(abs(np.random.default_rng().normal(np.mean(guess_bounds), 0.1*np.std(guess_bounds))))
else:
    x0 = None
x0 = comm.bcast(x0, root=0)
# choose population size for CMAES optimizer
popsize = 64
# set up CMAES optimizer
if rank == 0:
    es = cma.CMAEvolutionStrategy(x0, 1.0, {'bounds':bounds, 'maxiter':1000000, 'popsize':popsize})
# set up optimization loop
converged = False
counter = 0
while converged == False:
    # ask for solution an distribute
    solutions = np.empty((popsize, p.n_params), dtype=np.float64)
    if rank == 0:
        solutions = np.array(es.ask(), dtype=np.float64)
    for i in range(popsize):
        sol_temp = np.empty((p.n_params), dtype=np.float64)
        if rank == 0:
            sol_temp = solutions[i]
        comm.Bcast([sol_temp, MPI.DOUBLE], root=0)
        if rank !=0:
            solutions[i,:] = sol_temp[:]
    # compute fitness
    fit = [p.fitness(s)[0] for s in solutions]
    # tell fitness
    if rank == 0:
        es.tell(solutions, fit)
        if counter % 10 == 0:
            es.logger.add()
        es.disp()
        # check for convergence
        stop = es.stop()
        if len(stop) > 0:
            converged = True
            for k in stop:
                print (f"Stopping criterion: {k} = {str(stop[k])}")
    converged = comm.bcast(converged, root=0)
    # evaluate test/validation fitness and write to logfile
    if counter % 10 == 0:
        if rank == 0:
            params, fit_train = es.result[:2]
            params_test = np.array([params[i] for i in test_par_ids])
        else:
            fit_train = None
            params_test = np.empty((p_test.n_params))
        fit_train = comm.bcast(fit_train, root=0)
        comm.Bcast(params_test, root=0)
        fit_test = p_test.fitness(params_test)[0]
        if rank == 0:
            par_rms = np.sqrt(np.mean(params**2))
            with open("fitness.log", "a") as fitfile:
                fitfile.write(f"{fit_train:12.8f} {fit_test:12.8f} {par_rms:12.8f}\n")
    counter += 1
if rank == 0:
    print("PARAMS:")
    print(es.result[0])
