from qiskit.quantum_info import DensityMatrix, random_statevector, state_fidelity
from numpy import trace, matmul
from multiprocessing import Process
from time import time

from QEC import four_qubit_code, five_qubit_code, damp_err

NUM = 1000
num_qubit = 1
num_params = 20
damp_params = [i/(2*~-num_params) for i in range(num_params)]
fid_QEC5 = [0 for _ in range(len(damp_params))]
fid_SDP5 = [0 for _ in range(len(damp_params))]
fid_QEC4 = [0 for _ in range(len(damp_params))]
fid_SDP4 = [0 for _ in range(len(damp_params))]
fid_sing = [0 for _ in range(len(damp_params))]
states = [random_statevector(2).data for _ in range(NUM)]

def exec_QEC(rho, i):
    QEC_4 = four_qubit_code()
    try:
        fid_QEC4[i] += state_fidelity(DensityMatrix(rho), QEC_4.run(DensityMatrix(rho), damp_params[i]))/NUM
    except:
        fid_QEC4[i] += 0
    fid_SDP4[i] += QEC_4.run_SDP(damp_params[i], rho)/NUM
    
    QEC_5 = five_qubit_code()
    fid_QEC5[i] += state_fidelity(DensityMatrix(rho), QEC_5.run(DensityMatrix(rho), damp_params[i]))/NUM
    fid_SDP5[i] += QEC_5.run_SDP(damp_params[i], rho)/NUM
    
    E = damp_err(damp_params[i], 1)
    A = matmul(DensityMatrix(rho), E[0])
    for e in E[1:]:
        A += matmul(rho, e)
    fid_sing[i] += abs(trace(A))**2/NUM

iteration = 0
tot_iter = NUM
start = time()
for rho in states:
    print(f'\r{-~iteration}/{tot_iter}', end = '')
    if iteration:
        sec = (tot_iter-iteration)*past/iteration
        hrs = sec // 3600
        sec %= 3600
        mns = sec // 60
        sec %= 60
        print(f'\tETC {hrs:.0f} hours {mns:.0f} minuites {sec:.2f} seconds', end = '')
    iteration = -~iteration
    
    # Create multiple jobs
    jobs = []
    for i in range(num_params):
        job = Process(target = exec_QEC, args = (rho, i))
        job.start()
        jobs.append(job)

    # Wait for all jobs to finish
    for job in jobs:
        job.join()
        
    past = time() - start