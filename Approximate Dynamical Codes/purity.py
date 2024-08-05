from qiskit.quantum_info import DensityMatrix, random_statevector, random_density_matrix, purity
from numpy import trace, matmul, sqrt
from multiprocessing import Process, Manager
from matplotlib.pyplot import subplots, legend, savefig, show

from QEC import four_qubit_code, five_qubit_code, damp_err

NUM = 100
num_qubit = 1
num_params = 10
num_states = 1
iteration = 0
tot_iter = NUM
jobs_per_slot = 10
num_slots = num_params // jobs_per_slot
damp_params = [i/(2*~-num_params) for i in range(num_params)]
states = [DensityMatrix([[v/10, 0], [0, 1-v/10]]) for v in range(6)]#[random_density_matrix(2) for _ in range(num_states)]#[DensityMatrix(random_statevector(2)) for _ in range(num_states)]#[DensityMatrix([a/10, sqrt(1-(a/10)**2)]) for a in range(11)]
# num_states = len(states)

def exec_QEC(state, i):
    from os import getpid
    print(f'{i}: {getpid()}')
    QEC_4 = four_qubit_code()
    
    fid, enc, dec = QEC_4.run_SDP(damp_params[i], state, 1e-3)
    fid_SDP4[i] += fid/num_states
    print(i, damp_params[i], fid_SDP4[i], enc, dec)

_, ax = subplots(1, 1)
for state in states:
    print(f'{states.index(state)}: {state}')
    with Manager() as manager:
        fid_SDP4 = manager.list([0 for _ in range(len(damp_params))])
        if True:#for state in states:
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
    
        ax.plot(damp_params, fid_SDP4, label = f'Purity: {purity(state).real:0.2f}')
        ax.set_xlabel(r'Damping probability $(\gamma)$')
        ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
        legend()
        savefig(f'Purity.png')
