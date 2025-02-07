from qiskit.quantum_info import DensityMatrix, state_fidelity
from numpy import trace, matmul
from multiprocessing import Process, Manager

from QEC import four_qubit_code, five_qubit_code, damp_err, three_qubit_code

NUM = 100    # number of iteration for existing [4,1] and [[5,1,3]] codes
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
    Executes the QEC and SDP
    Arguments:
        state [<DensityMatrix>]: Density Matrix of the initial state
        i [int]: Index of the parameter in 'damp_params'
    Returns: None
    '''
    QEC_4 = four_qubit_code(num_err)
    QEC_5 = five_qubit_code(num_err)
    itr = NUM
    while(itr):
        try:
            # Execute [4,1] code. This code might fail with some probability.
            fid_QEC4[i] += state_fidelity(state, QEC_4.run(state, damp_params[i]))/NUM
            itr = ~-itr
            print(f'Running iteration... {itr} for {i}', end = '\r')
        except:
            print(f'Failed iteration... {itr} for {i}', end = '\r')
            continue
        # Execute [[5, 1, 3]] code
        fid_QEC5[i] += state_fidelity(state, QEC_5.run(state, damp_params[i]))/NUM

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
    fid_QEC4 = manager.list([0 for _ in range(len(damp_params))])
    fid_SDP4 = manager.list([0 for _ in range(len(damp_params))])
    fid_QEC5 = manager.list([0 for _ in range(len(damp_params))])
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

    print(fid_QEC5)
    print(fid_SDP5)
    print(fid_QEC4)
    print(fid_SDP4)
    print(fid_sing)

    from matplotlib.pyplot import subplots, legend, savefig, show


    _, ax = subplots(1, 1)
    ax.plot(damp_params, fid_QEC5, label = "[[5, 1, 3]] QEC")
    ax.plot(damp_params, fid_SDP5, label = "[[5, 1, 3]] see-saw")
    ax.plot(damp_params, fid_QEC4, label = "[4, 1] QEC")
    ax.plot(damp_params, fid_SDP4, label = "[4, 1] see-saw")
    ax.plot(damp_params, fid_sing, label = "Bare qubit")
    ax.set_xlabel(r'Damping probability $(\gamma)$')
    ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
    legend()
    show()
