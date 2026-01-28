import molsys
import numpy as np
from pathlib import Path
from mpi4py import MPI


# get MPI stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


def get_iter_range(ntotal):
    nchunk = ntotal // size
    nrest = ntotal % size
    iter_from = rank * nchunk
    iter_to = (rank + 1) * nchunk
    if rank < nrest:
        iter_from = rank * (nchunk + 1)
        iter_to = (rank + 1) * (nchunk + 1)
    else:
        iter_from = rank * nchunk + nrest
        iter_to = (rank + 1) * nchunk + nrest
    return iter_from, iter_to

class ChargeProblem:
    def __init__(self, mol, fit_EN=False, fit_sigma=False, fit_Jii=False, fit_q_base=False, static_Xij=False,
                 fit_Xij=False, fit_Xij_inter=False, use_q_ref=False, use_delta_q_ref=False, return_eig_J_min=False,
                 param_scalers_qeq = {}, param_scalers_acks2 = {}):
        self.model = mol.charge.model
        self.method = mol.charge.method
        self.mol = mol
        self.fit_EN = fit_EN
        self.fit_sigma = fit_sigma
        self.fit_Jii = fit_Jii
        self.fit_q_base = fit_q_base
        self.static_Xij = static_Xij
        self.fit_Xij = fit_Xij
        self.fit_Xij_inter = fit_Xij_inter
        self.use_q_ref = use_q_ref
        self.use_delta_q_ref = use_delta_q_ref
        self.return_eig_J_min = return_eig_J_min
        # these do nothing yet, best practice would be to read them in with the q_values somehow (TODO)
        self.delta_q_exceptions = []
        self.n_params = 0
        self.n_params_qeq = 0
        self.n_objectives = sum([self.use_q_ref, self.use_delta_q_ref])
        self.param_ids_qeq = {"sigma": 0, "Jii": 1, "EN": 2, "q_base": 3}
        # these are not ideal, a more elaborate rescaling scheme should be developed (TODO)
        self.param_scalers_qeq = {
            "sigma": 2.0, 
            "Jii": 1.0e3,
            "EN": 1.0e3,
            "q_base": 1.0
        }
        for k,v in param_scalers_qeq.items():
            self.param_scalers_qeq[k] = v
        if self.method == "acks2":
            self.param_ids_acks2 = {"Xij": 0, "Xij_2": 1, "Xij_inter": 0, "Xij_inter_2": 1}
            self.param_scalers_acks2 = {
                "Xij": 1.0,
                "Xij_2": 1.0,
                "Xij_inter": 1.0,
                "Xij_inter_2": 1.0e3
            }
            for k,v in param_scalers_acks2.items():
                self.param_scalers_acks2[k] = v
        # build the recalc sequence:
        self.recalc_sequence = []
        if self.fit_EN or self.use_delta_q_ref:
            self.recalc_sequence.append(self.model.setup_EN)
        if self.fit_sigma:
            self.recalc_sequence.append(self.model.setup_J)
        elif self.fit_Jii:
            self.recalc_sequence.append(self.model.reset_Jii)
        if self.fit_q_base:
            self.recalc_sequence.append(self.model.setup_q_base)
        if True in (self.fit_Xij, self.fit_Xij_inter):
            self.recalc_sequence.append(self.model.setup_X)
        self.recalc_sequence.append(self.model.solve)
        # calc once to have everything setup
        self.model.calc()
        return

    def get_param_table(self, fixes=[]):
        """
        return a list of strings refering to the parameters to be fitted
        call this function before any of get/set params to init the param_table
        """
        self.par_tab = []
        skip_list = fixes
        if self.method == "topoqeq":
            J_keys = self.model.params.keys()
        else:
            J_keys = self.model.Jij.params.keys()
        if self.method == "acks2":
            skip_list += [entry for value in self.model.Xij_equivs.values() for entry in value]
            X_keys = self.model.Xij.params.keys()
            utypes = self.model.utypes
            nbondmat = self.model.nbondmat
        if self.fit_sigma:
            for k in J_keys:
                if "sigma " + k not in skip_list:
                    self.par_tab.append("sigma " + k)
        if self.fit_Jii:
            for k in J_keys:
                if "Jii " + k not in skip_list:
                    self.par_tab.append("Jii " + k)
        if self.fit_EN:
            for k in J_keys:
                if "EN " + k not in skip_list:
                    self.par_tab.append("EN " + k)
        self.n_params_qeq = len(self.par_tab)
        if self.fit_Xij:
            for k in X_keys:
                k_split = k.split(":")
                i = utypes.index(k_split[0])
                j = utypes.index(k_split[1])
                nbonds = nbondmat[i, j]
                if nbonds > 0:
                    if k not in skip_list:
                        self.par_tab.append("Xij " + k)
                        if self.static_Xij == False:
                            self.par_tab.append("Xij_2 " + k)
        if self.fit_Xij_inter:
            for k in X_keys:
                k_split = k.split(":")
                i = utypes.index(k_split[0])
                j = utypes.index(k_split[1])
                nbonds = nbondmat[i, j]
                if nbonds == 0:
                    if k not in skip_list:
                        self.par_tab.append("Xij_inter " + k)
                        self.par_tab.append("Xij_inter_2 " + k)
        self.n_params = len(self.par_tab)
        return self.par_tab

    def get_params(self, scale=False):
        if self.method == "topoqeq":
            params = self.get_params_topoqeq(scale)
        else:
            params = self.get_params_qeq(scale)
            if self.method == "acks2":
                params += self.get_params_acks2(scale)
        return params

    def get_params_qeq(self, scale=False):
        ids = self.param_ids_qeq
        if scale == True:
            fac = {k : 1.0 / v for (k,v) in self.param_scalers_qeq.items()}
        else:
            fac = {k : 1.0  for (k,v) in self.param_scalers_qeq.items()}
        params = []
        for entry in self.par_tab[:self.n_params_qeq]:
            ptype, pname = entry.split()
            p = self.model.Jij.params[pname][ids[ptype]] * fac[ptype]
            params.append(p)
        return params

    def get_params_acks2(self, scale=False):
        ids = self.param_ids_acks2
        if scale == True:
            fac = {k : 1.0 / v for (k,v) in self.param_scalers_acks2.items()}
        else:
            fac = {k : 1.0  for (k,v) in self.param_scalers_acks2.items()}
        params = []
        for entry in self.par_tab[self.n_params_qeq:]:
            ptype, pname = entry.split()
            p = self.model.Xij.params[pname][ids[ptype]] * fac[ptype]
            params.append(p)
        return params

    def get_params_topoqeq(self, scale=False):
        ids = self.param_ids_qeq
        if scale == True:
            fac = {k : 1.0 / v for (k,v) in self.param_scalers_qeq.items()}
        else:
            fac = {k : 1.0  for (k,v) in self.param_scalers_qeq.items()}
        params = []
        for entry in self.par_tab:
            ptype, pname = entry.split()
            p = self.model.params[pname][ids[ptype]] * fac[ptype]
            params.append(p)
        return params

    def set_params(self, params):
        if self.method == "topoqeq":
            self.set_params_topoqeq(params)
        else:
            self.set_params_qeq(params[:self.n_params_qeq])
            if self.method == "acks2":
                self.set_params_acks2(params[self.n_params_qeq:])
        return

    def set_params_qeq(self, params):
        ids = self.param_ids_qeq
        fac = self.param_scalers_qeq
        for entry, param in zip(self.par_tab[:self.n_params_qeq], params):
            ptype, pname = entry.split()
            self.model.Jij.change_param(pname, ids[ptype], param * fac[ptype])
            if self.method == "qeq_cv":
                self.model.Jij_cv.change_param(pname, ids[ptype], param * fac[ptype])
                self.model.Jij_core.change_param(pname, ids[ptype], param * fac[ptype])
        self.model.Jij.setup_fparams()
        if self.method == "qeq_cv":
            self.model.Jij_cv.setup_fparams()
            self.model.Jij_core.setup_fparams()
        return

    def set_params_acks2(self, params):
        ids = self.param_ids_acks2
        fac = self.param_scalers_acks2
        equivs = self.model.Xij_equivs
        equiv_keys = equivs.keys()
        for entry, param in zip(self.par_tab[self.n_params_qeq:], params):
            ptype, pname = entry.split()
            self.model.Xij.change_param(pname, ids[ptype], param * fac[ptype])
            if pname in equiv_keys:
                for eq in equivs[pname]:
                    self.model.Xij.change_param(eq, ids[ptype], param * fac[ptype])
        self.model.Xij.setup_fparams()
        return
    
    def set_params_topoqeq(self, params):
        ids = self.param_ids_qeq
        fac = self.param_scalers_qeq
        for entry, param in zip(self.par_tab[:self.n_params_qeq], params):
            ptype, pname = entry.split()
            self.model.params[pname][ids[ptype]] = param * fac[ptype]
        return

    def read_refcharges(self, filename):
        with open(filename) as refchargefile:
            field_vals = []
            ref_charges = []
            lines = filter(None, (line.strip() for line in refchargefile))
            lines = (line.split() for line in lines if not line.startswith("#"))
            for line in lines:
                line_field = [
                    float(line[0]),
                    float(line[1]),
                    float(line[2]),
                ]
                line_charges = []
                for j, l in enumerate(line):
                    if j <= 2:
                        continue
                    line_charges.append(float(l))
                if line_field == [0.0, 0.0, 0.0]:
                    zero_ref_charges = [j for j in line_charges]
                else:
                    field_vals.append(line_field)
                    ref_charges.append(line_charges)
        self.field_ref_all = np.array(field_vals)
        self.q_ref_all = np.array(ref_charges)
        self.q_ref_zero = np.array(zero_ref_charges)
        # IS THIS A HACK?
        self.model.q_tot = np.around(np.sum(self.q_ref_zero))
        if self.q_ref_all.shape != (0,):
            self.delta_q_ref_all = self.q_ref_all - self.q_ref_zero
        return
    
    def recalc(self):
        for step in self.recalc_sequence:
            step()
        return
    
    def ssd_q(self, q_ref):
        delta_q = q_ref - self.model.q
        sd_q = delta_q * delta_q
        return np.sum(sd_q), sd_q.shape[0]

    def ssd_delta_q(self, q_zero, delta_q_ref, exceptions=[]):
        delta_q = self.model.q - q_zero
        delta_delta = delta_q_ref - delta_q
        delta_delta_f = np.delete(delta_delta, exceptions)
        sd_delta_q = delta_delta_f * delta_delta_f
        return np.sum(sd_delta_q), sd_delta_q.shape[0]

    def callback(self, par=None):
        ssd_list = []
        nd_list = []
        self.model.set_field([0.0, 0.0, 0.0])
        if par != None:
            self.set_params(par)
        self.recalc()
        q_zero_model = np.array([j for j in self.model.q])
        ssd_q, nd_q = self.ssd_q(self.q_ref_zero)
        if self.use_q_ref:
            ssd_list.append(ssd_q)
            nd_list.append(nd_q)
        if self.return_eig_J_min:
            eig_J_min = [np.min(np.linalg.eigvals(self.model.J_mat))]
        else:
            eig_J_min = []
        if self.use_delta_q_ref:
            ssd_delta_q = 0
            nd_delta_q = 0
            for i, field in enumerate(self.field_ref_all):
                self.model.set_field(field)
                self.recalc()
                ssd_i, nd_i = self.ssd_delta_q(q_zero_model, self.delta_q_ref_all[i])
                ssd_delta_q += ssd_i
                nd_delta_q += nd_i
            ssd_list.append(ssd_delta_q)
            nd_list.append(nd_delta_q)
            self.model.set_field([0.0, 0.0, 0.0])
        return ssd_list, nd_list, eig_J_min

    def fitness(self, par=None):
        ssd_list, nd_list, eig_list = self.callback(par)
        fit_list = [np.sqrt(ssd / nd) for (ssd, nd) in zip(ssd_list, nd_list)]
        #print("FIT " + " ".join("{:12.8f}".format(f) for f in fit_list))
        #print("EIGVALS " + " ".join("{:12.8f}".format(e) for e in eig_list))
        return fit_list
    

class MultiChargeProblem:
    def __init__(self):
        self.problems = []
        self.problem_names = []
        self.method = None
        self.n_problems = 0
        self.n_params = 0
        self.n_objectives = 0
        return
    
    def add_problem(self, problem, problem_name):
        self.problems.append(problem)
        self.problem_names.append(problem_name)
        self.n_problems += 1
        return
    
    def del_problem(self, problem_id):
        self.problems.pop(problem_id)
        self.problem_names.pop(problem_id)
        self.n_problems -= 1
        return
    
    def get_n_problems(self):
        return len(self.problems)
    
    def pool_qeq_params(self, mean_params=False):
        self.qeq_params = {}
        for problem in self.problems:
            # Gather qeq parameters
            params = {k : v[:] for k, v in problem.model.Jij.params.items()}
            nppatype = len(params[list(params.keys())[0]])
            for k, v in self.qeq_params.items():
                if k in params.keys():
                    params[k].extend(v)
                else:
                    params[k] = v
            self.qeq_params = params
        if mean_params == False:
            for k, v in self.qeq_params.items():
                self.qeq_params[k] = self.qeq_params[k][:nppatype]
        else:
            # Mean parameters from respective qeq.par files
            for k, v in self.qeq_params.items():
                new_v = np.array(v)
                new_v = new_v.reshape((len(v)//nppatype,nppatype))
                new_v = new_v.mean(axis=0)
                self.qeq_params[k] = new_v
                # redistribute meaned parameters to all targets
                for problem in self.problems:                
                    if k in problem.model.Jij.params.keys():
                        for i, nv in enumerate(new_v):
                            problem.model.Jij.change_param(k, i, nv)
                        if self.method == "qeq_cv":
                            for i, nv in enumerate(new_v):
                                problem.model.Jij_cv.change_param(k, i, nv)
                                problem.model.Jij_core.change_param(k, i, nv)
                    problem.model.Jij.calc_Jii_self()
                    if self.method == "qeq_cv":
                        problem.model.Jij_cv.calc_Jii_self()
        return
    
    def pool_acks2_params(self, mean_params=False):
        # Gather Xij parameters (in a particular way for easy access)
        Xij_dict = {}
        for i, problem in enumerate(self.problems):
            pname = self.problem_names[i]
            for k, v in problem.model.Xij.params.items():
                u1 = k.split(':')[0]
                u2 = k.split(':')[1]
                id1 = int(u1.split('%')[1]) - 1
                id2 = int(u2.split('%')[1]) - 1
                nbonds = problem.model.nbondmat[id1, id2]
                k_rev = u2 + ":" + u1
                Xij_dict[(pname, k)] = [u1, u2, nbonds, v]
                Xij_dict[(pname, k_rev)] = [u2, u1, nbonds, v]
        # prepare initial dictionary of target equivalencies using first problem
        self.Xij_params = {}
        self.Xij_problem_equivs = {}
        equivs_all = [problem.model.Xij_equivs for problem in self.problems]
        pname_0 = self.problem_names[0]
        for k, v in equivs_all[0].items():
            self.Xij_problem_equivs[(pname_0, k)] = []
            par = Xij_dict[(pname_0, k)][3]
            eq_pars = [Xij_dict[(pname_0, eq)][3] for eq in v]
            self.Xij_params[(pname_0, k)] = [[par] + eq_pars]
        # check for equivalencies with remaining problems
        for n in range(1, self.n_problems):
            pname_n = self.problem_names[n]
            for k, v in equivs_all[n].items():
                ui_1, ui_2, nbonds_i, par = Xij_dict[(pname_n, k)]
                eq_pars = [Xij_dict[(pname_n, eq)][3] for eq in v] 
                ai_1 = ui_1.split('%')[0]
                ai_2 = ui_2.split('%')[0]
                for j in self.Xij_problem_equivs.copy().keys():
                    uj_1, uj_2, nbonds_j = Xij_dict[j][:-1]
                    aj_1 = uj_1.split('%')[0]
                    aj_2 = uj_2.split('%')[0]
                    if [ai_1, ai_2] in [[aj_1, aj_2], [aj_2, aj_1]]:
                        if nbonds_i == nbonds_j:
                            self.Xij_problem_equivs[j].append((pname_n, k))
                            self.Xij_params[j] += [[par] + eq_pars]
                            break
                # add new entry if no equivalency is found
                else:
                    self.Xij_problem_equivs[(pname_n, k)] = []
                    self.Xij_params[(pname_n, k)] = [[par] + eq_pars]
        # gather keys to discriminate between Xij_intra/Xij_inter
        self.Xij_inter_keys = []
        self.Xij_intra_keys = []
        for k, v in self.Xij_problem_equivs.items():
            nbonds = Xij_dict[k][2]
            if nbonds == 0:
                self.Xij_inter_keys.append(k)
            else:
                self.Xij_intra_keys.append(k)
            if mean_params == False:
                self.Xij_params[k] = self.Xij_params[k][0][0]
            else:
                # take means and redistribute meaned parameters to all targets (this might not be the most efficient way of doing this)
                self.Xij_params[k] = np.array(self.Xij_params[k]).mean(axis=0)
                par = self.Xij_params[k]
                pname = k[0]
                pkey = k[1]
                problem_id = self.problem_names.index(pname)
                equivs = equivs_all[problem_id][pkey]
                problem = self.problems[problem_id]
                problem.model.Xij.change_param(pkey, 0, par[0])
                problem.model.Xij.change_param(pkey, 1, par[1])
                for eq in equivs:
                    problem.model.Xij.change_param(eq, 0, par[0])
                    problem.model.Xij.change_param(eq, 1, par[1])
                for peq in v:
                    pname = peq[0]
                    pkey = peq[1]
                    problem_id = self.problem_names.index(pname)
                    equivs = equivs_all[problem_id][pkey]
                    problem = self.problems[problem_id]
                    problem.model.Xij.change_param(pkey, 0, par[0])
                    problem.model.Xij.change_param(pkey, 1, par[1])
                    for eq in equivs:
                        problem.model.Xij.change_param(eq, 0, par[0])
                        problem.model.Xij.change_param(eq, 1, par[1])
        return

    def pool_topoqeq_params(self, mean_params=False):
        self.qeq_params = {}
        for problem in self.problems:
            # Gather qeq parameters
            params = {k : v[:] for k, v in problem.model.params.items()}
            nppatype = len(params[list(params.keys())[0]])
            for k, v in self.qeq_params.items():
                if k in params.keys():
                    params[k].extend(v)
                else:
                    params[k] = v
            self.qeq_params = params
        if mean_params == False:
            for k, v in self.qeq_params.items():
                self.qeq_params[k] = self.qeq_params[k][:nppatype]
        else:
            # Mean parameters from respective qeq.par files
            for k, v in self.qeq_params.items():
                new_v = np.array(v)
                new_v = new_v.reshape((len(v)//nppatype,nppatype))
                new_v = new_v.mean(axis=0)
                self.qeq_params[k] = new_v
                # redistribute meaned parameters to all targets
                for problem in self.problems:                
                    if k in problem.model.Jij.params.keys():
                        for i, nv in enumerate(new_v):
                            problem.model.params[k][i] = nv
        return
    
    def setup(self, mean_qeq_params=False, mean_acks2_params=False, fixes=[]):
        self.fit_keys = []
        self.fit_ids = [[] for i in range(self.n_problems)]
        # HACK(?): getting a bunch of stuff from first problem
        self.method = self.problems[0].method
        self.n_problems = self.get_n_problems()
        self.n_objectives = self.problems[0].n_objectives
        if self.method == "topoqeq":
            self.pool_topoqeq_params(mean_qeq_params)
        else:
            self.pool_qeq_params(mean_qeq_params)
            if self.method == "acks2":
                self.pool_acks2_params(mean_acks2_params)
        sorted_keys = sorted(self.qeq_params.keys())
        if self.problems[0].fit_sigma == True:
            self.fit_keys += ["sigma " + k for k in sorted_keys if "sigma " + k not in fixes]
        if self.problems[0].fit_Jii == True:
            self.fit_keys += ["Jii " + k for k in sorted_keys if "Jii " + k not in fixes]
        if self.problems[0].fit_EN == True:
            self.fit_keys += ["EN " + k for k in sorted_keys if "EN " + k not in fixes]
        if self.problems[0].fit_q_base == True:
            self.fit_keys += ["q_base " + k for k in sorted_keys if "q_base " + k not in fixes]
        if self.problems[0].fit_Xij == True:
            intra_params = {k : self.Xij_params[k] for k in self.Xij_intra_keys}
            #self.fit_keys += ["Xij " + str(k) for k in intra_params.keys() if "Xij " + str(k) not in fixes]
            #if self.problems[0].static_Xij == False:
            #    self.fit_keys += ["Xij_2 " + str(k) for k in intra_params.keys() if "Xij " + str(k) not in fixes]
            # HACK THIS FOR NOW
            self.fit_keys += ["Xij " + str(k) for k in intra_params.keys() if k[1] not in fixes]
            if self.problems[0].static_Xij == False:
                self.fit_keys += ["Xij_2 " + str(k) for k in intra_params.keys() if k[1] not in fixes]
        if self.problems[0].fit_Xij_inter == True:
            inter_params = {k : self.Xij_params[k] for k in self.Xij_inter_keys}
            #self.fit_keys += ["Xij_inter " + str(k) for k in inter_params.keys() if "Xij_inter " + str(k) not in fixes]
            #self.fit_keys += ["Xij_inter_2 " + str(k) for k in inter_params.keys() if "Xij_inter " + str(k) not in fixes]
            # HACK THIS FOR NOW
            self.fit_keys += ["Xij_inter " + str(k) for k in inter_params.keys() if k[1] not in fixes]
            self.fit_keys += ["Xij_inter_2 " + str(k) for k in inter_params.keys() if k[1] not in fixes]
        self.n_params = len(self.fit_keys)
        for i, (pname, problem) in enumerate(zip(self.problem_names, self.problems)):
            # get indices of self.fit_keys needed per par_tab
            for param in problem.par_tab:
                partype, k = param.split()
                if "Xij" not in partype:
                    self.fit_ids[i].append(self.fit_keys.index(param))
                else:
                    k_full = (pname,k)
                    param_full = partype + " " + str(k_full)
                    if k_full in self.Xij_problem_equivs.keys():
                        self.fit_ids[i].append(self.fit_keys.index(param_full))
                    else:
                        eq = next(k for (k, v) in self.Xij_problem_equivs.items() if k_full in v)
                        eqname = partype + " " + str(eq)
                        self.fit_ids[i].append(self.fit_keys.index(eqname))
        return

    def get_params(self, scale=False):
        params = self.n_params * [0.0]
        for i, problem in enumerate(self.problems):
            params_i = problem.get_params(scale)
            for idx, p_i in zip(self.fit_ids[i], params_i):
                params[idx] = p_i
        return params
    
    def set_params(self, params):
        for i, problem in enumerate(self.problems):
            newparams = [params[idx] for idx in self.fit_ids[i]]
            problem.set_params(newparams)
        return

    def fitness(self, par=None):
        if par is not None:
            self.set_params(par)
        ssd_list = []
        nd_list = []
        eig_list = []
        for problem in self.problems:
            ssd_i, nd_i, eig_J_min = problem.callback()
            ssd_list.append(ssd_i)
            nd_list.append(nd_i)
            eig_list += [e for e in eig_J_min]
        ssd_array = np.array(ssd_list)
        nd_array = np.array(nd_list)
        fit_single = np.sqrt(ssd_array/nd_array)
        #print("FIT SINGLE "+" ".join("{:12.8f}".format(f) for l in fit_single for f in l))
        fitness = np.sqrt(ssd_array.sum(axis=0)/nd_array.sum(axis=0))
        fit_list = [f for f in fitness]
        #print("FIT "+" ".join("{:12.8f}".format(f) for f in fit_list))
        if len(eig_list) > 0:
            print("EIGVALS "+" ".join("{:12.8f}".format(e) for e in eig_list))
        return fit_list
    
    def write_charges(self):
        for t, problem in enumerate(self.problems):
            field_array = problem.field_ref_all
            ref_array = problem.q_ref_all
            zero_ref = problem.q_ref_zero
            if self.problems[0].use_delta_q_ref:
                value_array = np.zeros([field_array.shape[0]+1,field_array.shape[1]+ref_array.shape[1]])
            else:
                value_array = np.zeros([1,zero_ref.shape[0]+3])
            problem.recalc()
            q = problem.model.q
            value_array[0,3:] = q[:]
            print("REF: ", zero_ref, "TOTAL:", zero_ref.sum())
            print("Q:   ", q, "TOTAL:", q.sum())
            if self.problems[0].use_delta_q_ref:
                for i, field in enumerate(field_array):
                    problem.model.set_field(field)
                    problem.recalc()
                    q = problem.model.q
                    value_array[i+1,3:] = q[:]
                    print("REF: ", ref_array[i])
                    print("Q:   ", q)
                value_array[1:,:3] = field_array
            with open("outcharges_{:d}.dat".format(t), 'w') as outfile:
                for v in value_array:
                    outfile.write(" ".join("{:<10.8f}".format(q) for q in v))
                    outfile.write("\n")
        return

class MultiChargeProblem_MPI:
    def __init__(self):
        self.problems = []
        self.problem_names = []
        self.method = None
        self.n_problems = 0
        self.n_params = 0
        self.n_objectives = 0
        return
    
    def add_problem(self, problem, problem_name):
        self.problems.append(problem)
        self.problem_names.append(problem_name)
        self.n_problems += 1
        return
    
    def del_problem(self, problem_id):
        self.problems.pop(problem_id)
        self.problem_names.pop(problem_id)
        self.n_problems -= 1
        return
    
    def get_n_problems(self):
        return len(self.problems)

    def compute_n_problems_all(self):
        self.n_problems_all = comm.allreduce(self.n_problems)
        return self.n_problems_all

    def pool_qeq_params(self, mean_params=False):
        self.qeq_params = {}
        for problem in self.problems:
            # Gather qeq parameters
            params = {k : v[:] for k, v in problem.model.Jij.params.items()}
            nppatype = len(params[list(params.keys())[0]])
            for k, v in self.qeq_params.items():
                if k in params.keys():
                    params[k].extend(v)
                else:
                    params[k] = v
            self.qeq_params = params
        print(rank, self.qeq_params.keys())
        qeq_params_all = comm.gather(self.qeq_params, root=0)
        if rank == 0:
            self.qeq_params_all = qeq_params_all[0]
            for i in qeq_params_all[1:]:
                for k, v in i.items():
                    if k in self.qeq_params_all.keys():
                        self.qeq_params_all[k].extend(v)
                    else:
                        self.qeq_params_all[k] = v
        else:
            self.qeq_params_all = None
        self.qeq_params_all = comm.bcast(self.qeq_params_all, root=0)
        print(rank, self.qeq_params_all.keys())
        if mean_params == False:
            for k, v in self.qeq_params.items():
                self.qeq_params[k] = self.qeq_params_all[k][:nppatype]
        else:
            raise NotImplementedError
        return
    
    def pool_acks2_params(self, mean_params=False):
        # Gather Xij parameters (in a particular way for easy access)
        Xij_dict = {}
        for i, problem in enumerate(self.problems):
            pname = self.problem_names[i]
            for k, v in problem.model.Xij.params.items():
                u1 = k.split(':')[0]
                u2 = k.split(':')[1]
                id1 = int(u1.split('%')[1]) - 1
                id2 = int(u2.split('%')[1]) - 1
                nbonds = problem.model.nbondmat[id1, id2]
                k_rev = u2 + ":" + u1
                Xij_dict[(pname, k)] = [u1, u2, nbonds, v]
                Xij_dict[(pname, k_rev)] = [u2, u1, nbonds, v]
        Xij_dict_all = comm.gather(Xij_dict, root=0)
        if rank == 0:
            Xij_dict_all_tmp = Xij_dict_all[0]
            for i in Xij_dict_all[1:]:
                for k, v in i.items():
                    if k in Xij_dict_all_tmp.keys():
                        Xij_dict_all_tmp[k].extend(v)
                    else:
                        Xij_dict_all_tmp[k] = v
            Xij_dict_all = Xij_dict_all_tmp
        else:
            Xij_dict_all = None
        Xij_dict_all = comm.bcast(Xij_dict_all, root=0)
        # prepare initial dictionary of target equivalencies using first problem
        self.Xij_params = {}
        self.Xij_problem_equivs = {}
        equivs_all = [problem.model.Xij_equivs for problem in self.problems]
        pname_0 = self.problem_names[0]
        for k, v in equivs_all[0].items():
            self.Xij_problem_equivs[(pname_0, k)] = []
            par = Xij_dict[(pname_0, k)][3]
            eq_pars = [Xij_dict[(pname_0, eq)][3] for eq in v]
            self.Xij_params[(pname_0, k)] = [[par] + eq_pars]
        # check for equivalencies with remaining problems
        for n in range(1, self.n_problems):
            pname_n = self.problem_names[n]
            for k, v in equivs_all[n].items():
                ui_1, ui_2, nbonds_i, par = Xij_dict[(pname_n, k)]
                eq_pars = [Xij_dict[(pname_n, eq)][3] for eq in v] 
                ai_1 = ui_1.split('%')[0]
                ai_2 = ui_2.split('%')[0]
                for j in self.Xij_problem_equivs.copy().keys():
                    uj_1, uj_2, nbonds_j = Xij_dict[j][:-1]
                    aj_1 = uj_1.split('%')[0]
                    aj_2 = uj_2.split('%')[0]
                    if [ai_1, ai_2] in [[aj_1, aj_2], [aj_2, aj_1]]:
                        if nbonds_i == nbonds_j:
                            self.Xij_problem_equivs[j].append((pname_n, k))
                            self.Xij_params[j] += [[par] + eq_pars]
                            break
                # add new entry if no equivalency is found
                else:
                    self.Xij_problem_equivs[(pname_n, k)] = []
                    self.Xij_params[(pname_n, k)] = [[par] + eq_pars]
        Xij_params_all = comm.gather(self.Xij_params, root=0)
        Xij_problem_equivs_all = comm.gather(self.Xij_problem_equivs, root=0)
        if rank == 0:
            self.Xij_params_all = Xij_params_all[0]
            self.Xij_problem_equivs_all = Xij_problem_equivs_all[0]
            for Xij_problem_equivs_i, Xij_params_i in zip(Xij_problem_equivs_all[1:], Xij_params_all[1:]):
                for k, v in Xij_problem_equivs_i.items():
                    ui_1, ui_2, nbonds_i, par = Xij_dict_all[k]
                    ai_1 = ui_1.split('%')[0]
                    ai_2 = ui_2.split('%')[0]
                    for j in self.Xij_problem_equivs_all.keys():
                        uj_1, uj_2, nbonds_j = Xij_dict_all[j][:-1]
                        aj_1 = uj_1.split('%')[0]
                        aj_2 = uj_2.split('%')[0]
                        if [ai_1, ai_2] in [[aj_1, aj_2], [aj_2, aj_1]]:
                            if nbonds_i == nbonds_j:
                                self.Xij_problem_equivs_all[j].append(k)
                                for v_i in v:
                                    self.Xij_problem_equivs_all[j].append(v_i)
                                self.Xij_params_all[j] += Xij_params_i[k]
                                break
            # gather keys to discriminate between Xij_intra/Xij_inter
            self.Xij_inter_keys = []
            self.Xij_intra_keys = []
            for k, v in self.Xij_problem_equivs_all.items():
                nbonds = Xij_dict_all[k][2]
                if nbonds == 0:
                    self.Xij_inter_keys.append(k)
                else:
                    self.Xij_intra_keys.append(k)
        else:
            self.Xij_params_all = None
            self.Xij_problem_equivs_all = None
            self.Xij_intra_keys = None
            self.Xij_inter_keys = None
        self.Xij_params_all = comm.bcast(self.Xij_params_all, root=0)
        self.Xij_problem_equivs_all = comm.bcast(self.Xij_problem_equivs_all, root=0)
        self.Xij_intra_keys = comm.bcast(self.Xij_intra_keys, root=0)
        self.Xij_inter_keys = comm.bcast(self.Xij_inter_keys, root=0)
        for k, v in self.Xij_problem_equivs.items():
            if mean_params == False:
                if k in self.Xij_params_all.keys():
                    self.Xij_params[k] = self.Xij_params_all[k][0][0]
                else:
                    eq = next(k_eq for (k_eq, v_eq) in self.Xij_problem_equivs_all.items() if k in v_eq)
                    self.Xij_params[k] = self.Xij_params_all[eq][0][0]
            else:
                raise NotImplementedError
        return

    def pool_topoqeq_params(self, mean_params=False):
        self.qeq_params = {}
        for problem in self.problems:
            # Gather qeq parameters
            params = {k : v[:] for k, v in problem.model.params.items()}
            nppatype = len(params[list(params.keys())[0]])
            for k, v in self.qeq_params.items():
                if k in params.keys():
                    params[k].extend(v)
                else:
                    params[k] = v
            self.qeq_params = params
        qeq_params_all = comm.gather(self.qeq_params, root=0)
        if rank == 0:
            self.qeq_params_all = qeq_params_all[0]
            for i in qeq_params_all[1:]:
                for k, v in i.items():
                    if k in self.qeq_params_all.keys():
                        self.qeq_params_all[k].extend(v)
                    else:
                        self.qeq_params_all[k] = v
        else:
            self.qeq_params_all = None
        self.qeq_params_all = comm.bcast(self.qeq_params_all, root=0)
        if mean_params == False:
            for k, v in self.qeq_params.items():
                self.qeq_params[k] = self.qeq_params_all[k][:nppatype]
        else:
            raise NotImplementedError
        return
    
    def setup(self, mean_qeq_params=False, mean_acks2_params=False, fixes=[]):
        self.compute_n_problems_all()
        self.fit_keys = []
        self.fit_ids = [[] for i in range(self.n_problems)]
        # HACK(?): getting a bunch of stuff from first problem
        self.method = self.problems[0].method
        self.n_problems = self.get_n_problems()
        self.n_objectives = self.problems[0].n_objectives
        if self.method == "topoqeq":
            self.pool_topoqeq_params(mean_qeq_params)
        else:
            self.pool_qeq_params(mean_qeq_params)
            if self.method == "acks2":
                self.pool_acks2_params(mean_acks2_params)
        sorted_keys = sorted(self.qeq_params_all.keys())
        if self.problems[0].fit_sigma == True:
            self.fit_keys += ["sigma " + k for k in sorted_keys if "sigma " + k not in fixes]
        if self.problems[0].fit_Jii == True:
            self.fit_keys += ["Jii " + k for k in sorted_keys if "Jii " + k not in fixes]
        if self.problems[0].fit_EN == True:
            self.fit_keys += ["EN " + k for k in sorted_keys if "EN " + k not in fixes]
        if self.problems[0].fit_q_base == True:
            self.fit_keys += ["q_base " + k for k in sorted_keys if "q_base " + k not in fixes]
        if self.problems[0].fit_Xij == True:
            intra_params = {k : self.Xij_params_all[k] for k in self.Xij_intra_keys}
            self.fit_keys += ["Xij " + str(k) for k in intra_params.keys() if "Xij " + str(k) not in fixes]
            if self.problems[0].static_Xij == False:
                self.fit_keys += ["Xij_2 " + str(k) for k in intra_params.keys() if "Xij " + str(k) not in fixes]
        if self.problems[0].fit_Xij_inter == True:
            inter_params = {k : self.Xij_params_all[k] for k in self.Xij_inter_keys}
            self.fit_keys += ["Xij_inter " + str(k) for k in inter_params.keys() if "Xij_inter " + str(k) not in fixes]
            self.fit_keys += ["Xij_inter_2 " + str(k) for k in inter_params.keys() if "Xij_inter " + str(k) not in fixes]
        self.n_params = len(self.fit_keys)
        for i, (pname, problem) in enumerate(zip(self.problem_names, self.problems)):
            # get indices of self.fit_keys needed per par_tab
            for param in problem.par_tab:
                partype, k = param.split()
                if "Xij" not in partype:
                    self.fit_ids[i].append(self.fit_keys.index(param))
                else:
                    k_full = (pname,k)
                    param_full = partype + " " + str(k_full)
                    if k_full in self.Xij_problem_equivs_all.keys():
                        self.fit_ids[i].append(self.fit_keys.index(param_full))
                    else:
                        eq = next(k for (k, v) in self.Xij_problem_equivs_all.items() if k_full in v)
                        eqname = partype + " " + str(eq)
                        self.fit_ids[i].append(self.fit_keys.index(eqname))
        return

    def get_params(self, scale=False):
        params = self.n_params * [0.0]
        for i, problem in enumerate(self.problems):
            params_i = problem.get_params(scale)
            for idx, p_i in zip(self.fit_ids[i], params_i):
                params[idx] = p_i
        return params
    
    def set_params(self, params):
        for i, problem in enumerate(self.problems):
            newparams = [params[idx] for idx in self.fit_ids[i]]
            problem.set_params(newparams)
        return

    def fitness(self, par=None):
        if par is not None:
            self.set_params(par)
        ssd_list = []
        nd_list = []
        eig_list = []
        for problem in self.problems:
            ssd_i, nd_i, eig_J_min = problem.callback()
            ssd_list.append(ssd_i)
            nd_list.append(nd_i)
            eig_list += [e for e in eig_J_min]
        ssd_array = np.array(ssd_list)
        nd_array = np.array(nd_list)
        sssd = ssd_array.sum(axis=0)
        snd = nd_array.sum(axis=0)
        sssd_all = comm.allreduce(sssd)
        snd_all = comm.allreduce(snd)
        fitness = np.sqrt(sssd_all/snd_all)
        fit_list = [f for f in fitness]
        #print("FIT "+" ".join("{:12.8f}".format(f) for f in fit_list))
        if len(eig_list) > 0:
            print("EIGVALS "+" ".join("{:12.8f}".format(e) for e in eig_list))
        return fit_list
    
    def write_charges(self):
        for t, problem in enumerate(self.problems):
            field_array = problem.field_ref_all
            ref_array = problem.q_ref_all
            zero_ref = problem.q_ref_zero
            if self.problems[0].use_delta_q_ref:
                value_array = np.zeros([field_array.shape[0]+1,field_array.shape[1]+ref_array.shape[1]])
            else:
                value_array = np.zeros([1,zero_ref.shape[0]+3])
            problem.recalc()
            q = problem.model.q
            value_array[0,3:] = q[:]
            print("REF: ", zero_ref, "TOTAL:", zero_ref.sum())
            print("Q:   ", q, "TOTAL:", q.sum())
            if self.problems[0].use_delta_q_ref:
                for i, field in enumerate(field_array):
                    problem.model.set_field(field)
                    problem.recalc()
                    q = problem.model.q
                    value_array[i+1,3:] = q[:]
                    print("REF: ", ref_array[i])
                    print("Q:   ", q)
                value_array[1:,:3] = field_array
            with open("outcharges_{:d}.dat".format(t), 'w') as outfile:
                for v in value_array:
                    outfile.write(" ".join("{:<10.8f}".format(q) for q in v))
                    outfile.write("\n")
        return


def generate_problem(molfile, fixes=[], refchargefile=None, modelargs=[], modelkwargs={}, problemargs=[], problemkwargs={}):
    m = molsys.mol.from_file(molfile)
    m.addon("charge")
    m.charge.set_model(*modelargs, **modelkwargs)
    problem = ChargeProblem(m, *problemargs, **problemkwargs)
    problem.get_param_table(fixes = fixes)
    if refchargefile != None:
        problem.read_refcharges(refchargefile)
    return problem


def generate_multiproblem(targetdir, fixes=[], master_qeq_parfile=None, master_acks2_parfile=None, 
                          modelargs=[], modelkwargs={}, problemargs=[], problemkwargs={}):
    multiproblem = MultiChargeProblem()
    targets = Path(targetdir)
    fixes_copy = fixes.copy()
    for d in targets.iterdir():
        molfile = str(list(d.glob("*.mfpx"))[0])
        # this is not entirely thought through...
        mean_qeq_params = False
        mean_acks2_params = False
        #assign qeq parameters if given
        qeq_parfile = d / "qeq.par"
        if master_qeq_parfile:
            modelkwargs["qeq_parfile"] = master_qeq_parfile
        elif Path(qeq_parfile).is_file():
            modelkwargs["qeq_parfile"] = qeq_parfile
            mean_qeq_params = True
        #assign acks2 parameters if given
        acks2_parfile = d / "acks2.par"
        if master_acks2_parfile:
            modelkwargs["acks2_parfile"] = master_acks2_parfile 
        if Path(acks2_parfile).is_file():
            modelkwargs["acks2_parfile"] = acks2_parfile
            mean_acks2_params = True
        # assign reference charges if given
        refchargefile = d / "charges.in"
        if Path(refchargefile).is_file() == False:
            refchargefile = None
        # generate and add problem
        problem = generate_problem(molfile, fixes, refchargefile, modelargs, modelkwargs, problemargs, problemkwargs)
        multiproblem.add_problem(problem, d.name)
        print(f"target {d.name} added")
    multiproblem.setup(mean_qeq_params, mean_acks2_params, fixes=fixes)
    # NOTE: This should be cleaned up at some point:
    param_ids_qeq = {"sigma": 0, "Jii": 1, "EN": 2, "q_base": 3}
    # REMOVE THIS FOR NOW
    #for i in fixes_copy:
    #    ptype, atype = i.split()
    #    param = multiproblem.qeq_params[atype][param_ids_qeq[ptype]]
    #    print(f"Fixing Parameter {i} to {param}")
    return multiproblem

def generate_multiproblem_mpi(targetdir, fixes=[], master_qeq_parfile=None, master_acks2_parfile=None, 
                          modelargs=[], modelkwargs={}, problemargs=[], problemkwargs={}):
    multiproblem = MultiChargeProblem_MPI()
    targets = Path(targetdir)
    l_targets = [*targets.iterdir()]
    n_targets = len(l_targets)
    fixes_copy = fixes.copy()
    iter_from, iter_to = get_iter_range(n_targets)
    natoms_arr = np.zeros(n_targets, dtype=np.int32)
    for i in range(iter_from, iter_to):
        molfile = list(l_targets[i].glob("*.mfpx"))[0]
        with open(molfile, 'r') as mfile:
            line = mfile.readline()
            firstchar =  line[0]
            while firstchar == '#':
                line = mfile.readline()
                firstchar = line[0]
            else:
                natoms_arr[i] = int(line)
    natoms_all = np.zeros(n_targets, dtype=np.int32)
    comm.Reduce(natoms_arr, natoms_all, op=MPI.SUM, root=0)
    if rank == 0:
        sorted_ids = np.argsort(natoms_all)
        targets_sorted = [l_targets[i] for i in sorted_ids]
    else:
        targets_sorted = None
    targets_sorted = comm.bcast(targets_sorted, root=0)
    for t in range(rank, n_targets, size):
        d = targets_sorted[t]
        molfile = str(list(d.glob("*.mfpx"))[0])
        # this is not entirely thought through...
        mean_qeq_params = False
        mean_acks2_params = False
        #assign qeq parameters if given
        qeq_parfile = d / "qeq.par"
        if master_qeq_parfile:
            modelkwargs["qeq_parfile"] = master_qeq_parfile
        elif Path(qeq_parfile).is_file():
            modelkwargs["qeq_parfile"] = qeq_parfile
            mean_qeq_params = True
        #assign acks2 parameters if given
        acks2_parfile = d / "acks2.par"
        if master_acks2_parfile:
            modelkwargs["acks2_parfile"] = master_acks2_parfile 
        if Path(acks2_parfile).is_file():
            modelkwargs["acks2_parfile"] = acks2_parfile
            mean_acks2_params = True
        # assign reference charges if given
        refchargefile = d / "charges.in"
        if Path(refchargefile).is_file() == False:
            refchargefile = None
        # generate and add problem
        problem = generate_problem(molfile, fixes, refchargefile, modelargs, modelkwargs, problemargs, problemkwargs)
        multiproblem.add_problem(problem, d.name)
        print(f"target {d.name} added")
    multiproblem.setup(mean_qeq_params, mean_acks2_params, fixes=fixes)
    # NOTE: This should be cleaned up at some point:
    param_ids_qeq = {"sigma": 0, "Jii": 1, "EN": 2, "q_base": 3}
    for i in fixes_copy:
        ptype, atype = i.split()
        param = multiproblem.qeq_params[atype][param_ids_qeq[ptype]]
        print(f"Fixing Parameter {i} to {param}")
    return multiproblem
