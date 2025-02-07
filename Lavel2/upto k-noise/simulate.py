from qiskit.quantum_info import DensityMatrix, partial_trace, Operator, state_fidelity, Choi, Pauli
from qiskit import QuantumCircuit
from math import sqrt, comb
from pickle import load
from numpy import matrix, array, trace, kron, float64, dtype, zeros, ndarray
from multiprocessing import Process, Lock, shared_memory

def damp_err(gamma, n, m):
    '''This method produces noise operators for a non-Markovian amplitude damping channel with damping strength gamma.
    Arguments:
		gamma [float]: Damping probability
		n [int]: Number of total qubit
		m [int]: Number of erroneous qubit
	Returns:
		Kraus operators corresponding to the channel
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, kron, sqrt, zeros
    from itertools import combinations, product
    
    # The operators from the both parts are similar. But the second part has a negative eigen value.
    mu = gamma/2	# parameter for the first part
    lamda = (gamma-mu)/(1-mu)	# parameter for the second part
    
    # Operators for the first part
    _E0 = [eye(2), zeros((2,2))]
    _E0[0][1][1] = sqrt(1-mu)
    _E0[1][0][1] = sqrt(mu)
    
    # Operators for the second part
    _E1 = [eye(2), zeros((2,2))]
    _E1[0][1][1] = sqrt(1-lamda)
    _E1[1][0][1] = sqrt(lamda)
    
    E0 = []
    E1 = []
    
    id_ind = list(combinations(range(n), n-m))	# combinations of qubits without noise
    # Repeat noise oparators m times
    E0_ = list(product(_E0, repeat = m))
    E1_ = list(product(_E1, repeat = m))
    for i in id_ind:
        for j in range(len(E0_)):
            # Builds noise operators for multiple qubits
            err_ops0 = E0_[j]
            err_ops1 = E1_[j]
            E_0 = eye(1)/sqrt(len(id_ind))
            E_1 = eye(1)
            for k in range(n):
                # Apply noise on appropriate qubits
                if k in i:
                    # Apply identity operation on qubit k
                    E_0 = kron(E_0, eye(2))
                    E_1 = kron(E_1, eye(2))
                else:
                    # Apply noise operation on qubit k
                    E_0 = kron(E_0, err_ops0[0])
                    err_ops0 = err_ops0[1:]
                    E_1 = kron(E_1, err_ops1[0])
                    err_ops1 = err_ops1[1:]
            E0.append(E_0.copy())
            E1.append(E_1.copy())
    return E0, E1
    
def cal_N01(noise_ops0, noise_ops1):
    '''Calculates N0 and N1 as in Algorithm 1.
    Arguments:
    	noise_ops0 [list]: List of noise operators for the first part
    	noise_ops1 [list]: List of noise operators for the second part
	Returns: None
    '''
    from numpy import outer, zeros, kron
    
    nops0 = []
    nops1 = []
    for a in range(2**num_qubit):
        nops0.append([])
        nops1.append([])
        for b in range(2**num_qubit):
            nops0[a].append([])
            nops1[a].append([])
            for noise_op0, noise_op1 in zip(noise_ops0, noise_ops1):
                nops0[a][b].append(outer(noise_op0.conjugate()[:,a], noise_op0[:,b]))
                nops1[a][b].append(outer(noise_op1.conjugate()[:,a], noise_op1[:,b]))
    return nops0, nops1

def cal_R1(state):
    '''Calculates R1 as in Algorithm 1.
    Argument:
    	state [<DensityMatrix>]: Density matrix corresponding to the initial logical state (for entanglement fidelity, it is completely mixed state)
	Returns: None
    '''
    from numpy import outer, kron, zeros
    
    nops = []
    for s in range(2**num_qubit):
        nops.append([])
        for t in range(2**num_qubit):
            nops[s].append([])
            nops[s][t] = outer(state.data[:,s], state.data.conjugate()[:,t])
    return nops

num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]
state = DensityMatrix([[0.5, 0], [0, 0.5]])
num_qubit = int(input('Enter number of physical qubits in a logical qubit: '))
num_error = int(input('Enter number of erroneous qubits: '))
if num_qubit == 1:
	# 1-qubit
	enc = Choi(Operator([[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]])).data
	chk = Choi(Operator([[0, 1], [1, 0]])).data
	dec = Choi(Operator([[1/sqrt(2), 1/sqrt(2)], [-1/sqrt(2), 1/sqrt(2)]])).data
if num_qubit == 2:
	# 2-qubit
	qc = QuantumCircuit(2)
	qc.h(-1)
	qc.cx(-1, 0)
	enc = Choi(Operator(qc)).data
	chk = Choi(Operator([[sqrt(0.5), 0, sqrt(0.5), sqrt(0.5)], [0, 1, 0, 0], [sqrt(0.5), 0, -sqrt(0.5), -sqrt(0.5)], [0, 0, 0, 0]])).data
	dec = Choi(Operator([[1, sqrt(0.5), 0, 0], [0, 0, 0, 0], [0, -sqrt(0.5), 1, 0], [0, 0, 0, 1]])).data
if num_qubit == 3:
	# 3-qubit
	qc = QuantumCircuit(3)
	qc.h(-1)
	qc.cx(-1, 0)
	qc.cx(-1, 1)
	enc = Choi(Operator(qc)).data
	chk = Choi(Operator([[1, 0, 0, 0, 0, 0, 0, 0],
										   [0, 1, 0, 0, 0, 0, 0, 0],
										   [0, 0, 1, 0, 0, 0, 0, 0],
										   [0, 0, 0, 1, 0, 0, 0, 0],
										   [0, 0, 0, 0, 1, 0.5, 0.5, 1], 
										   [0, 0, 0, 0, 0, 0.5, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0.5, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0]])).data
	dec = Choi(Operator([[sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5)],
										   [0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0],
										   [sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5)],
										   [0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0]])).data
if num_qubit == 4:
	# 4-qubit
	qc = QuantumCircuit(4)
	qc.h(-1)
	qc.cx(-1, 0)
	qc.cx(-1, 1)
	qc.cx(-1, 2)
	enc = Choi(Operator(qc)).data
	chk = Choi(Operator([[1, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0, 0],
										   [0, 1, 0, 1/3, 0, 1/3,    0,    0, 0, 1/3,    0,    0,    0,    0,    0, 0],
										   [0, 0, 1, 1/3, 0,    0, 1/3,    0, 0,    0, 1/3,    0,    0,    0,    0, 0],
										   [0, 0, 0, 1/3, 0,    0,    0, 0.5, 0,    0,   0, 0.5,    0,    0,    0, 0],
										   [0, 0, 0,    0, 1, 1/3, 1/3,    0, 0,    0,    0,    0, 1/3,    0,    0, 0], 
										   [0, 0, 0,    0, 0, 1/3,    0, 0.5, 0,    0,    0,    0,    0, 0.5,    0, 0],
										   [0, 0, 0,    0, 0,    0, 1/3, 0.5, 0,    0,    0,    0,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0, 0.5, 0,    0,    0,    0,    0,    0,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 1, 1/3, 1/3,     0, 1/3,    0,    0, 1],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0, 1/3,    0, 0.5,    0, 1/3,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0, 1/3, 0.5,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0, 0.5,    0,    0,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0, 1/3, 1/3, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0, 1/3,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0, 0]])).data
	dec = Choi(Operator([[sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5)],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5), -sqrt(0.5)],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])).data

def fid_cal(lock, a, fid):
    '''This method creates the objective function as in Step 6 of Algorithm 1.
    Arguments:
    	lock [<Lock>]: Lock used for multiprocess scheduling to access shared memory
    	chk [list]: List of check operators in Choi form
    	a [int]: An index specified in Step 6 of Algorithm 1
    	obj [<shared_memory>]: Shared memory for objective function of the SDP
	Returns: None
    '''
    from numpy import zeros, array, trace, kron, complex128
    from math import comb
    from os import getpid
    from time import time
    fid_L = 0
    for b in range(2**num_qubit):
        start = time()
        for c in range(2**num_qubit):
            for d in range(2**num_qubit):
                for s in [0, 2**(~-num_qubit)]:	# For other values, the corresponding R1 entry is 0
                    for t in [0, 2**(~-num_qubit)]:	# For other values, the corresponding R1 entry is 0
                        for i in range(2**num_error*comb(num_qubit, num_error)):
                        	for j in range(2**num_error):
                        		# The relation between i and j ensures that the noise is applied on the same qubit in the both part
                        		if (i%2**num_error)&j:
                        			continue
                        		j += i-i%2**num_error
                        		tra = 0
                        		for x in range(2**num_qubit):
                        			for y in range(2**num_qubit):
                        				tra += array(chk)[x*2**num_qubit+d][y*2**num_qubit+c]*N0[a][b][i][y][x]
                        		fid_L += array(enc)[t*2**num_qubit+b][s*2**num_qubit+a]*trace(array(dec)@kron(N1[c][d][j], R1[s][t]))*tra
    lock.acquire()	# locks the shared memory
    fid += fid_L.real	# updates the shared memory
    lock.release()	# release the lock

anc = [1] + [0] * ~-2**(~-num_qubit)
ancilla = DensityMatrix(anc)
state = ancilla.expand(state)
R1 = cal_R1(state)
fidL = []
for gamma in damp_params:
    noise_ops0, noise_ops1 = damp_err(gamma, num_qubit, num_error)
    N0, N1 = cal_N01(noise_ops0, noise_ops1)
    d_size = dtype(float64).itemsize
    shm = shared_memory.SharedMemory(create = True, size = d_size, name = f'fidelity{num_qubit}{num_error}')
    fid = ndarray(shape = (1,), dtype = float64, buffer = shm.buf)
    fid[:] = zeros(1, float64)
    jobs = []
    lock = Lock()
    for a in range(2**num_qubit):
    	job = Process(target = fid_cal, args = (lock, a, fid))
    	job.start()
    	jobs.append(job)
    for job in jobs:
    	job.join()
    fidL.append(fid[0].copy())
    shm.close()
    shm.unlink()
    print(fidL[-1], end = ', ')
print()
