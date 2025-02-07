from qiskit.quantum_info import DensityMatrix, state_fidelity
from numpy import trace, matmul
from multiprocessing import Process, Manager

from QEC import four_qubit_code, five_qubit_code, damp_err, three_qubit_code

NUM = 100
num_err = int(input('Enter number of erroneous qubits: '))
num_params = 10
iteration = 0
tot_iter = NUM
jobs_per_slot = 20    # maximum number of simultaneous processes during multiprocessing.
num_slots = num_params // jobs_per_slot
damp_params = [i/(2*~-num_params) for i in range(num_params)]
state = DensityMatrix([[0.5, 0], [0, 0.5]])

def exec_QEC(state, i):
    '''
    Executes the see-saw SDP
    Arguments:
        state [<DensityMatrix>]: Density Matrix of the initial state
        i [int]: Index of the parameter in 'damp_params'
    Returns: None
    '''
    QEC_4 = four_qubit_code(num_err)
    QEC_5 = five_qubit_code(num_err)

    # Perform SDPs
    fid_SDP4[i] += QEC_4.run_SDP(damp_params[i], state)
    fid_SDP5[i] += QEC_5.run_SDP(damp_params[i], state)

    # Calculating entanglement fidelity without encoding
    E = damp_err(damp_params[i], 1, 1)
    if not E:
    	pass
    A = abs(trace(matmul(state.data, E[0])))**2
    for e in E[1:]:
        A += abs(trace(matmul(state.data, e)))**2
    fid_sing[i] += A

with Manager() as manager:
    fid_SDP4 = manager.list([0 for _ in range(len(damp_params))])
    fid_SDP5 = manager.list([0 for _ in range(len(damp_params))])
    fid_sing = manager.list([0 for _ in range(len(damp_params))])
    init = 0
    # Will execute <jobs_per_slot> jobs at a time
    for slot in range(num_slots):
        # Create multiple jobs
        jobs = []
        for i in range(init, init+jobs_per_slot):
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

    print(fid_SDP5)
    print(fid_SDP4)
    print(fid_sing)
