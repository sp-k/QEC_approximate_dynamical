from qiskit.quantum_info import DensityMatrix, state_fidelity
from numpy import trace, matmul
from multiprocessing import Process, Manager

from QEC import four_qubit_code, five_qubit_code, damp_err

NUM = 1000
num_qubit = 1
num_params = 100
iteration = 0
tot_iter = NUM
jobs_per_slot = 10
num_slots = num_params // jobs_per_slot
damp_params = [i/(2*~-num_params) for i in range(num_params)]
states = [{0: 0.5, 1: 0.5}]

def exec_QEC(state, i):
    rho = []
    s_v = [0, 0]
    for sv, prob in state.items():
    	s_v[sv] = prob
    	rho.append(s_v)
    
    QEC_4 = four_qubit_code()
    QEC_5 = five_qubit_code()
    itr = NUM
    while(itr):
        try:
            fid_QEC4[i] += state_fidelity(DensityMatrix(rho), QEC_4.run(DensityMatrix(rho), damp_params[i]))/NUM
            itr = ~-itr
        except:
            continue
        fid_QEC5[i] += state_fidelity(DensityMatrix(rho), QEC_5.run(DensityMatrix(rho), damp_params[i]))/NUM
    
    fid_SDP4[i] += QEC_4.run_SDP(damp_params[i], state)
    fid_SDP5[i] += QEC_5.run_SDP(damp_params[i], state)
    
    E = damp_err(damp_params[i], 1)
    A = abs(trace(matmul(DensityMatrix(rho), E[0])))**2
    for e in E[1:]:
        A += abs(trace(matmul(rho, e)))**2
    fid_sing[i] += A

with Manager() as manager:
    fid_QEC4 = manager.list([0 for _ in range(len(damp_params))])
    fid_SDP4 = manager.list([0 for _ in range(len(damp_params))])
    fid_QEC5 = manager.list([0 for _ in range(len(damp_params))])
    fid_SDP5 = manager.list([0 for _ in range(len(damp_params))])
    fid_sing = manager.list([0 for _ in range(len(damp_params))])
    for state in states:
        init = 0
        # Will execute 10 jobs at a time
        for slot in range(num_slots):
            # Create multiple jobs
            jobs = []
            for i in range(init, init+10):
                job = Process(target = exec_QEC, args = (state, i))
                job.start()
                jobs.append(job)
    
            # Wait for all jobs to finish
            for job in jobs:
                job.join()
            init += jobs_per_slot
            
        jobs = []
        for i in range(init, num_params):
            job = Process(target = exec_QEC, args = (state, i))
            job.start()
            jobs.append(job)
    
        # Wait for all jobs to finish
        for job in jobs:
            job.join()


    from matplotlib.pyplot import subplots, legend, savefig, show


    _, ax = subplots(1, 1)
    ax.plot(damp_params, fid_QEC5, label = "[[5, 1, 3]]")
    ax.plot(damp_params, fid_SDP5, label = "[[5, 1, 3]] SDP")
    ax.plot(damp_params, fid_QEC4, label = "[[4, 1, 2]]")
    ax.plot(damp_params, fid_SDP4, label = "[[4, 1, 2]] SDP")
    ax.plot(damp_params, fid_sing, label = "Single qubit")
    ax.set_xlabel(r'Damping probability $(\gamma)$')
    ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
    legend()
    show()
