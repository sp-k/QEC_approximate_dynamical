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
        
    if m > n:
    	raise ValueError(f'Number of physical qubits {n} is less than the number of erroneous qubits {m}')
    
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
    
    for j in range(m+1):
        id_ind = list(combinations(range(n), n-j))	# combinations of qubits without noise
        for i in id_ind:
            E_0 = eye(1)
            E_1 = eye(1)
            for k in range(n):
                # Apply noise on appropriate qubits
                if k in i:
                    # Apply no-error operation on qubit k
                    E_0 = kron(E_0, _E0[0])
                    E_1 = kron(E_1, _E1[0])
                else:
                    # Apply error noise operation on qubit k
                    E_0 = kron(E_0, _E0[1])
                    E_1 = kron(E_1, _E1[1])
            E0.append(E_0.copy())
            E1.append(E_1.copy())
    return E0, E1

class QECC_seesaw:
    def __init__(self, enc_op, chk_op, num_qubit, num_error):
        '''
        Arguments:
        	enc_op [<Choi>]: Choi form of initial encoding operator
        	chk_op [list of <Choi>]: Choi form of initial check operators
        	num_qubit [int]: Number of physical qubits in a logical qubit
        	num_error [int]: Number of erroneous qubits
    	Returns: None
        '''
        self.enc_op = enc_op
        self.chk_op = chk_op
        self.num_qubit = num_qubit
        self.num_error = num_error
        self.fidelity = 10		# initialized as 10 to ensure convergence
        pass
    
    def cal_N01(self, noise_ops0, noise_ops1):
        '''Calculates N0 and N1 as in Algorithm 1.
        Arguments:
        	noise_ops0 [list]: List of noise operators for the first part
        	noise_ops1 [list]: List of noise operators for the second part
    	Returns: None
        '''
        from numpy import outer, zeros, kron
        
        nops0 = []
        nops1 = []
        for a in range(2**self.num_qubit):
            nops0.append([])
            nops1.append([])
            for b in range(2**self.num_qubit):
                nops0[a].append([])
                nops1[a].append([])
                for noise_op0, noise_op1 in zip(noise_ops0, noise_ops1):
                    nops0[a][b].append(outer(noise_op0.conjugate()[:,a], noise_op0[:,b]))
                    nops1[a][b].append(outer(noise_op1.conjugate()[:,a], noise_op1[:,b]))
#         print(len(nops0[a][b]))
#         print(len(nops1[a][b]))
        self.N0 = nops0
        self.N1 = nops1
#         print(_)
        pass
    
    def cal_R1(self, state):
        '''Calculates R1 as in Algorithm 1.
        Argument:
        	state [<DensityMatrix>]: Density matrix corresponding to the initial logical state (for entanglement fidelity, it is completely mixed state)
    	Returns: None
        '''
        from numpy import outer, kron, zeros
        
        nops = []
        for s in range(2**self.num_qubit):
            nops.append([])
            for t in range(2**self.num_qubit):
                nops[s].append([])
                nops[s][t] = outer(state.data[:,s], state.data.conjugate()[:,t])
        self.R1 = nops
        '''for s in range(2**self.num_qubit):
            for t in range(2**self.num_qubit):
                if (self.R1[s][t] != zeros(self.R1[s][t].shape)).any():
                	print(s, t, 2**(~-self.num_qubit))'''
#         print(len(nops), len(nops[0]), nops)
        pass
    
    @staticmethod
    def get_cd(num_qubits, c, d):
        '''This method creates elementary matrices (i.e., a matrix which has all zero elements with a single 1 at (c, d)).
        Arguments:
        	num_qubits [int]: Denotes size of the matrix (2**num_qubits X 2**num_qubits)
        	(c, d) [(int, int)]: Possition of 1
    	Returns:
    		The elementery matrix
        '''
        from numpy import zeros
        
        cd = zeros((2**num_qubits, 2**num_qubits))
        cd[c][d] = 1
        return cd
    
    def obj_dec(self, lock, chk, a, obj):
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
        print(f'{a}: {getpid()}')	# prints PID of the current process
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# local memory to calculate the objective function
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                        for t in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                            for n0 in self.N0[a][b]:
                            	for n1 in self.N1[c][d]:
                            		tra = 0
                            		for x in range(2**self.num_qubit):
		                                for y in range(2**self.num_qubit):
		                                    tra += array(chk)[x*2**self.num_qubit+d][y*2**self.num_qubit+c]*n0[y][x]
                            		objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*tra*kron(n1, self.R1[s][t])
            print(f'[Dec] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()	# locks the shared memory
        obj += objL	# updates the shared memory
        lock.release()	# release the lock
    
    def obj_chk(self, lock, dec, a, obj):
        '''This method creates the objective function as in Step 8 of Algorithm 1.
        Arguments:
        	lock [<Lock>]: Lock used for multiprocess scheduling to access shared memory
        	dec [list]: List of decoding operators in Choi form
        	a [int]: An index specified in Step 8 of Algorithm 1
        	obj [<shared_memory>]: Shared memory for objective function of the SDP
    	Returns: None
        '''
        from numpy import zeros, array, trace, kron, complex128
        from math import comb
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')	# prints PID of the current process
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# local memory to calculate the objective function
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                        for t in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                            for n0 in self.N0[a][b]:
                            	for n1 in self.N1[c][d]:
                            		objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*trace(array(dec)@kron(n1, self.R1[s][t]))*kron(n0, self.get_cd(self.num_qubit, c, d))
            print(f'[Chk] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()	# locks the shared memory
        obj += objL	# updates the shared memory
        lock.release()	# release the lock
    
    def obj_enc(self, lock, a, obj):
        '''This method creates the objective function as in Step 10 of Algorithm 1.
        Arguments:
        	lock [<Lock>]: Lock used for multiprocess scheduling to access shared memory
        	a [int]: An index specified in Step 10 of Algorithm 1
        	obj [<shared_memory>]: Shared memory for objective function of the SDP
    	Returns: None
        '''
        from numpy import zeros, kron, trace, kron, complex128
        from math import comb
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')	# prints PID of the current process
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# local memory to calculate the objective function
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                        for t in [0, 2**(~-self.num_qubit)]:	# For other values, the corresponding R1 entry is 0
                            for n0 in self.N0[a][b]:
                            	for n1 in self.N1[c][d]:
                            		for m in range(len(self.chk_op)):
		                                mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
		                                mat[s*2**self.num_qubit+a][t*2**self.num_qubit+b] = 1
		                                tra = 0
		                                for x in range(2**self.num_qubit):
		                                    for y in range(2**self.num_qubit):
		                                        tra += self.chk_op[m][x*2**self.num_qubit+d][y*2**self.num_qubit+c]*n0[y][x]
		                                objL += tra*trace(self.dec_op[m]@kron(n1, self.R1[s][t]))*mat
            print(f'[Enc] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()	# locks the shared memory
        obj += objL	# updates the shared memory
        lock.release()	# release the lock
    
    def run(self, _ATOL = 1e-2):
        '''This method performs the SDPs.
        Arguments:
        Returns: None
        '''
        from cvxpy import partial_trace, trace, real, Variable, Problem, Maximize
        from numpy import eye, zeros, complex128, dtype, prod, ndarray
        from multiprocessing import Process, Manager, Lock, shared_memory
        from qiskit.quantum_info import state_fidelity, DensityMatrix
        
        itr = 0
        while(True):
            itr += 1
            # SDP for Decoding
            print(f'Running Iteration {itr}...')

            # Define variable
            variable = []
            for _ in range(len(self.chk_op)):
                variable.append(Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian = True))

            # Define constraints
            constraints = [var >> 0 for var in variable] + [partial_trace(var, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit) for var in variable]

            # Define objective function
            Obj = []
            for chk in self.chk_op:
                d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))	# memory size for the objective function
                shm = shared_memory.SharedMemory(create = True, size = d_size, name = f'object{self.num_qubit}{self.num_error}')	# creates shared memory
                obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)	# builds the objective function with the shared memory
                obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# initializes the objective function with 0
                jobs = []
                lock = Lock()
                for a in range(2**self.num_qubit):
                    # Perform parallel computation with multiprocessing
                    job = Process(target = self.obj_dec, args = (lock, chk, a, obj))	# fork a process
                    job.start()	# start the process
                    jobs.append(job)	# put it in a list to access later
                for job in jobs:
                    job.join()	# wait untill all processes are completed
                Obj.append(obj.copy())	# copy the shared memory in a normal memory
                shm.close()	# close access of shared memory
                shm.unlink()	# release the shared memory
            # Step 6 of Algorithm 1
            objective = variable[0]@Obj[0]
            for var, obj in zip(variable[1:], Obj[1:]):
                objective += var@obj
            objective = real(trace(objective))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()	# build and solve the problem
            self.dec_op = [var.value for var in variable]	# decoding operators as the solutions of the SDP
            print(self.fidelity, fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:	# convergence checking
                break
            self.fidelity = fidelity	# update the fidelity with the SDP solution
            
            # SDP for Checking
#             print(f'Iteration {itr}: Running SDP for checking... {fidelity} {self.fidelity}', end = ', ')

            # Define variable
            variable = []
            for _ in range(len(self.chk_op)):
                variable.append(Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian = True))

            # Define constraints
            chk = variable[0]
            for var in variable[1:]:
                chk += var
            constraints = [var >> 0 for var in variable] + [partial_trace(chk, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            Obj = []
            for dec in self.dec_op:
                d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))	# memory size for the objective function
                shm = shared_memory.SharedMemory(create = True, size = d_size, name = f'object{self.num_qubit}{self.num_error}')	# creates shared memory
                obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)	# builds the objective function with the shared memory
                obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# initializes the objective function with 0
                jobs = []
                lock = Lock()
                for a in range(2**self.num_qubit):
                    # Perform parallel computation with multiprocessing
                    job = Process(target = self.obj_chk, args = (lock, dec, a, obj))	# fork a process
                    job.start()	# start the process
                    jobs.append(job)	# put it in a list to access later
                for job in jobs:
                    job.join()	# wait untill all processes are completed
                Obj.append(obj.copy())	# copy the shared memory in a normal memory
                shm.close()	# close access of shared memory
                shm.unlink()	# release the shared memory
            # Step 8 of Algorithm 1
            objective = variable[0]@Obj[0]
            for var, obj in zip(variable[1:], Obj[1:]):
                objective += var@obj
            objective = real(trace(objective))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()	# build and solve the problem
            self.chk_op = [var.value for var in variable]	# check operators as the solutions of the SDP
            print(self.fidelity, fidelity)
#             self.chk_op, dm_final = self.normalize('check', gamma)
#             fidelity = state_fidelity(DensityMatrix([[0.5, 0], [0, 0.5]]), dm_final, False)
#             print(fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:	# convergence checking
                break
            self.fidelity = fidelity	# update the fidelity with the SDP solution
                
            # SDP for Encoding
#             print(f'Iteration {itr}: Running SDP for encoding... {fidelity} {self.fidelity}', end = ', ')

            # Define variable
            variable = Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian = True)

            # Define constraints
            constraints = [variable >> 0, partial_trace(variable, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))	# memory size for the objective function
            shm = shared_memory.SharedMemory(create = True, size = d_size, name = f'object{self.num_qubit}{self.num_error}')	# creates shared memory
            Obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)	# builds the objective function with the shared memory
            Obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)	# initializes the objective function with 0
            jobs = []
            lock = Lock()
            for a in range(2**self.num_qubit):
                # Perform parallel computation with multiprocessing
                job = Process(target = self.obj_enc, args = (lock, a, Obj))	# fork a process
                job.start()	# start the process
                jobs.append(job)	# put it in a list to access later
            for job in jobs:
                job.join()	# wait untill all processes are completed
            objective = real(trace(variable@Obj.copy()))	# Step 10 of Algorithm 1
            shm.close()	# close access of shared memory
            shm.unlink()	# release the shared memory
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()	# build and solve the problem
            self.enc_op = variable.value	# encoding operator as the solutions of the SDP
            print(self.fidelity, fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:	# convergence checking
                break
            self.fidelity = fidelity	# update the fidelity with the SDP solution
        self.fidelity = fidelity
        print(fidelity)
        pass
    
class n_qubit_code:
    def __init__(self, n, m, q):
        '''
        Arguments:
        	n [int]: Number of total qubits
        	m [int]: Number of memory for adaptivity
        	q [int]: Number of erroneous qubits
    	Returns: None
        '''
        self.num_qubit = n
        self.num_mem = m
        self.num_error = q
        pass
    
    def run_SDP(self, gamma, state, _ATOL = 1e-2):
        '''Executes the QEC with SDP.
        Arguments:
        	gamma [float]: Damping streangth
        	state [list or numpy array]: Density matrix
        Returns:
        	A tuple containing optimal fidelity, encoding, check and decoding operators
        '''
        
        # Expand state with ancilla qubits
        from qiskit.quantum_info import DensityMatrix, Choi
        anc = [1] + [0] * ~-2**(~-self.num_qubit)
        ancilla = DensityMatrix(anc)
        state = ancilla.expand(state)
        
        # Get the noise operators
        noise_ops0, noise_ops1 = damp_err(gamma, self.num_qubit, self. num_error)
        
        # Perform seesaw
#         print('Initiating see-saw...')
        seesaw = QECC_seesaw(Choi(self._init_enc(self.num_qubit)), [Choi(chk) for chk in self._init_chk(self.num_qubit, self.num_mem)], self.num_qubit, self.num_error)
#         print('N0')
        seesaw.cal_N01(noise_ops0, noise_ops1)
#         print('N1')
        seesaw.cal_R1(state)
#         print('Runing see-saw...')
        seesaw.run(_ATOL)
        return (seesaw.fidelity, seesaw.enc_op, seesaw.chk_op, seesaw.dec_op)
    
    @staticmethod
    def _init_enc(num_qubits):
        '''Initializes the encoding operation
        '''
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Pauli
        from numpy import eye
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        if num_qubits != 1:
            qc.cx([0, 0], [1, -1])
            qc.cx(-1, -2)
        return Operator(qc)
        for i in range(num_qubits-1):
            qc.cx(-1, i)
        return Operator(qc)
    
    @staticmethod
    def _init_chk(num_qubits, num_mem):
        '''Initializes the check operations
        '''
        from numpy import eye, kron, zeros, sqrt
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Pauli, Choi
        
        chk = []
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits-1):
            qc.cx(-1, i)
            
        match (num_qubits, num_mem):
            case (1, 1):
                return [Operator(qc).compose(Pauli('X'))]
            case (1, 2):
                return [Operator(qc).compose(Pauli('I'))/sqrt(2), Operator(qc).compose(Pauli('I'))/sqrt(2)]
            case (1, 3):
                return [Operator(qc).compose(Pauli('X'))/3, Operator(qc).compose(Pauli('X'))/3, Operator(qc).compose(Pauli('X'))/3]
            case (2, 1):
                qc.h(-1)
                return [Operator(qc).compose(Pauli('II'))]
            case (2, 2):
                qc.h(0)
                return [Operator(qc).compose(Pauli('XX'))/sqrt(2), Operator(qc).compose(Pauli('XX'))/sqrt(2)]
                zero = zeros((2, 2))
                M0 = zero.copy()
                M0[0][0] = 1
                M1 = zero.copy()
                M1[1][1] = 1
                chk.append(Operator(qc).compose(kron(eye(2), M0)))
                chk.append(Operator(qc).compose(kron(eye(2), M1)))
                return chk
            case (2, 3):
                zero = zeros((4, 4))
                M00 = zero.copy()
                M00[0][0] = 1
                M01 = zero.copy()
                M01[1][1] = 1
                M1 = zeros((2, 2))
                M1[1][1] = 1
                chk.append(Operator(qc).compose(M00))
                chk.append(Operator(qc).compose(M01))
                chk.append(Operator(qc).compose(kron(M1, eye(2))))
                zero = zeros((2, 2))
                M0 = zero.copy()
                M0[0][0] = 1
                M1 = zero.copy()
                M1[1][1] = 1
                chk.append(Operator(qc).compose(kron(eye(2), M0)))
                chk.append(Operator(qc).compose(kron(eye(2), M1)))
                chk.append(Operator(qc))
                return chk
            case (3, 1):
                return [Operator(qc).compose(Pauli('III'))]
            case (3, 2):
                zero = zeros((2, 2))
                M0 = zero.copy()
                M0[0][0] = 1
                M1 = zero.copy()
                M1[1][1] = 1
                chk.append(Operator(qc).compose(kron(eye(4), M0)))
                chk.append(Operator(qc).compose(kron(eye(4), M1)))
                return chk
            case (3, 3):
                zero = zeros((4, 4))
                M00 = zero.copy()
                M00[0][0] = 1
                M01 = zero.copy()
                M01[1][1] = 1
                M1 = zeros((2, 2))
                M1[1][1] = 1
                chk.append(Operator(qc).compose(kron(eye(2), M00)))
                chk.append(Operator(qc).compose(kron(eye(2), M01)))
                chk.append(Operator(qc).compose(kron(eye(2), kron(M1, eye(2)))))
                return chk
            case (4, 1):
                return [Operator(qc).compose(Pauli('IIII'))]
            case (5, 1):
                return [Operator(qc).compose(Pauli('IIIII'))]
