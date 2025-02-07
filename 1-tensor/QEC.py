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

	if m > n:	# Number of erroneous qubit is more than the number physical qubit
		return 0

	from itertools import combinations, product
	from numpy import eye, zeros, sqrt, kron

	# Noise operator for single qubit
	_E0 = [eye(2), zeros((2,2))]
	_E0[0][1][1] = (1-gamma)**0.5
	_E0[1][0][1] = gamma**0.5

	E0 = [] # stores n-qubit noise operators

	id_ind = list(combinations(range(n), n-m))	# combinations of qubits without noise
	# Repeat noise oparators m times
	E0_ = list(product(_E0, repeat = m))
	for i in id_ind:
		for j in range(len(E0_)):
		    # Builds noise operators for multiple qubits
		    err_ops0 = E0_[j]
		    E_0 = eye(1)/sqrt(len(id_ind))
		    for k in range(n):
		        # Apply noise on appropriate qubits
		        if k in i:
		            # Apply identity operation on qubit k
		            E_0 = kron(E_0, eye(2))
		        else:
		            # Apply noise operation on qubit k
		            E_0 = kron(E_0, err_ops0[0])
		            err_ops0 = err_ops0[1:]
		    E0.append(E_0.copy())
	return E0

def damp_err_all(gamma, n):
    '''This method produces all-qubit noise operators. Equivalent to 'damp_err(gamma, n, n)'
    gamma [float]: Damping probability
    n [int]: Number of qubit
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, zeros, kron, sqrt
    
    _E = [eye(2), zeros((2, 2))]
    _E[0][1][1] = sqrt(1-gamma)
    _E[1][0][1] = sqrt(gamma)
    
    E = _E.copy()
    for m in range(1, n):
        E_ = []
        for i in _E:
            for j in E:
                E_.append(kron(i, j))
        E = E_.copy()
    return E

def damp_err_single(gamma, n):
    '''This method produces 1-qubit noise operators. Equivalent to 'damp_err(gamma, n, 1)'
    gamma [float]: Damping probability
    n [int]: Number of qubit
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, zeros, kron, sqrt
    
    _E = [eye(2), zeros((2, 2))]
    _E[0][1][1] = sqrt(1-gamma)
    _E[1][0][1] = sqrt(gamma)
    
    E = []
    for j in range(n):
        E.append(kron(eye(2**j), kron(_E[0]/sqrt(n), eye(2**(n-j-1)))))
        E.append(kron(eye(2**j), kron(_E[1]/sqrt(n), eye(2**(n-j-1)))))
    return E

def vec(A):
    '''This method vectorizes a matrix'''
    from numpy import matrix, concatenate, array
    return matrix(concatenate(array(A), axis = None)).conjugate()

class QECC_seesaw:
    def __init__(self, enc_op, num_qubit, num_err):
        '''
        Arguments:
        	enc_op [<Choi>]: Choi form of initial encoding operator
        	num_qubit [int]: Number of physical qubits in a logical qubit
        	num_error [int]: Number of erroneous qubits
    	Returns: None
        '''
        self.enc_op = enc_op
        self.num_qubit = num_qubit
        self.num_error = min(num_qubit, num_err)
        self.fidelity = 10	# initialized as 10 to ensure convergence
        pass
    
    def cal_N0(self, noise_ops):
        '''Calculates N0 as in Algorithm 4.
        Arguments:
        	noise_ops [list]: List of noise operators
    	Returns: None
        '''
        from numpy import outer, zeros, kron
        
        nops = []
        for i in range(2**self.num_qubit):
            nops.append([])
            for j in range(2**self.num_qubit):
                nops[i].append([])
                nops[i][j] = outer(noise_ops[0].conjugate()[:,i], noise_ops[0][:,j])
                for noise_op in noise_ops[1:]:
                    nops[i][j] += outer(noise_op.conjugate()[:,i], noise_op[:,j])
        self.N0 = nops
        pass
    
    def cal_R1(self, state):
        '''Calculates R as in Algorithm 4.
        Argument:
        	state [<DensityMatrix>]: Density matrix corresponding to the initial logical state (for entanglement fidelity, it is completely mixed state)
    	Returns: None
        '''
        from numpy import outer, kron
        
        nops = []
        for k in range(2**self.num_qubit):
            nops.append([])
            for l in range(2**self.num_qubit):
                nops[k].append([])
                nops[k][l] = outer(state.data[:,k], state.data.conjugate()[:,l])
        self.R1 = nops
#         print(len(nops), len(nops[0]), nops)
        pass
    
    def obj_dec(self):
        '''This method creates the objective function as in Step 5 of Algorithm 4.
        Arguments:
		
    	Returns: None
        '''
        from numpy import zeros, array, trace, kron, complex128
        from time import time
        from os import getpid
        obj = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        start = time()
        for i in range(2**self.num_qubit):
            for j in range(2**self.num_qubit):
                for k in range(2**self.num_qubit):
                    for l in range(2**self.num_qubit):
                        obj += array(self.enc_op)[l*2**self.num_qubit+j][k*2**self.num_qubit+i]*kron(self.N0[i][j], self.R1[k][l])
        print(f'[({getpid()}) Dec] Time taken: {time()-start}')
        return obj
    
    def obj_enc(self):
        '''This method creates the objective function as in Step 7 of Algorithm 4.
        Arguments:
		
    	Returns: None
        '''
        from numpy import zeros, kron, trace, kron, complex128
        from time import time
        from os import getpid
        obj = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        start = time()
        for i in range(2**self.num_qubit):
            for j in range(2**self.num_qubit):
                for k in range(2**self.num_qubit):
                    for l in range(2**self.num_qubit):
                        mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                        mat[k*2**self.num_qubit+i][l*2**self.num_qubit+j] = 1
                        obj += trace(self.dec_op@kron(self.N0[i][j], self.R1[k][l]))*mat
        print(f'[({getpid()}) Enc] Time taken: {time()-start}')
        return obj
    
    def run(self, gamma, _ATOL = 1e-2):
        '''This method performs the SDPs.
        Arguments:
		gamma [float]: Damping strength
  		_ATOL: Tolerance in fidelity convergence
        Returns: None
        '''
        from cvxpy import partial_trace, trace, real, Variable, Problem, Maximize
        from numpy import eye, zeros, complex128, dtype, prod, ndarray
        from qiskit.quantum_info import state_fidelity, DensityMatrix
        
        itr = 0
        while(True):
            itr += 1
            # SDP for Decoding
            print(f'Running Iteration {itr}...')

            # Define variable
            variable = Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian = True)

            # Define constraints
            constraints = [variable >> 0, partial_trace(variable, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            obj = self.obj_dec()
            objective = real(trace(variable@obj.copy()))
            
            # Solve SDP
#             print('Solving SDP')
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.dec_op = variable.value
            print(self.fidelity, fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:
                break
            self.fidelity = fidelity
                
            # SDP for Encoding
#             print(f'Iteration {itr}: Running SDP for encoding... {fidelity} {self.fidelity}', end = ', ')

            # Define variable
            variable = Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian = True)

            # Define constraints
            constraints = [variable >> 0, partial_trace(variable, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            Obj = self.obj_enc()
            objective = real(trace(variable@Obj.copy()))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.enc_op = variable.value
            print(self.fidelity, fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:
                break
            self.fidelity = fidelity
        self.fidelity = fidelity
        print(fidelity)
        pass
        
class five_qubit_code:
    def __init__(self, num_err):
        self.num_qubit = 5
        self.num_error = min(self.num_qubit, num_err)
        self.enc_op = self._get_enc_op()    #Encoding operator
        self.cor_op = None    #Correction operator
        pass
    
    def _get_enc_op(self):
        '''Creates Encoding operator'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from qiskit.circuit.library import MCMT
        
        # Initialize the circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Apply encoding operations
        qc.x(4)
        qc.cx(4, 0)
        qc.x(4)
        qc.h(4)
        qc.cx(4, 0)
        qc.h(3)
        qc.cx([3, 3], [2, 1])
        qc.x(0)
        qc.cx(0, 2)
        qc.cz(0, 2)
        qc.x(0)
        qc.cx(4, 0)
        qc.x(0)
        qc.append(MCMT('z', 2, 1), [0, 1, 3])
        qc.x(0)
        qc.cx(4, 0)
        qc.h([3, 0])
        qc.cx([3, 0], [2, 1])
        qc.swap([0, 3], [1, 4])
      
        return Operator(qc)
    
    def _get_cor_op(self, synd_res):
        '''Creates correction operation
        synd_res [str]: Syndrome measurement result
        '''
        
        from qiskit.quantum_info import Operator, Pauli
        self.cor_op = Operator(Pauli(self.codebook[int(synd_res, 2)]))
        pass
    
    def run_SDP(self, gamma, state, _ATOL = 1e-2):
        '''Executes the QEC with SDP
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Optimal fidelity
        '''
        
        # Expand state with ancilla qubits
        from qiskit.quantum_info import DensityMatrix, Choi
        anc = [1] + [0] * ~-2**(~-self.num_qubit)
        ancilla = DensityMatrix(anc)
        state = ancilla.expand(state)
        noise_ops = damp_err(gamma, self.num_qubit, self.num_error)
        
        # Perform seesaw
#         print('Initiating see-saw...')
        seesaw = QECC_seesaw(Choi(self.enc_op), self.num_qubit, self.num_error)
#         print('N0')
        seesaw.cal_N0(noise_ops)
#         print('N1')
        seesaw.cal_R1(state)
#         print('Runing see-saw...')
        seesaw.run(_ATOL)
        return seesaw.fidelity

class four_qubit_code:
    def __init__(self, num_err):
        self.num_qubit = 4
        self.num_error = min(self.num_qubit, num_err)
        self.enc_op = self._get_enc_op()    #Encoding operator
        self.cor_op = None    #Correction operator
        pass
    
    def _get_enc_op(self):
        '''Creates Encoding operator'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        # Initialize the circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Apply encoding operations
        qc.h(1)
        qc.cx(1, 2)
        qc.cx([2, 1], [3, 0])
      
        return Operator(qc)
    
    def _get_cor_op(self, gamma):
        '''Creates correction operation
        gamma [float]: Damping probability
        '''
        
        from numpy import arctan, arccos, pi
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        self.cor_op = []
        for i in range(3):
            if not i:
                theta = arctan((1-gamma)**2)
                qc = QuantumCircuit(2)
                qc.cx(1, 0)
                qc.ry(2*theta, 1)
                qc.cry(pi/2-2*theta, 0, 1)
                self.cor_op.append(Operator(qc))
            elif not ~-i:
                theta = arccos(1-gamma)
                qc = QuantumCircuit(3)
                qc.x(2)
                qc.cry(theta, 2, 0)
                self.cor_op.append(Operator(qc))
            elif not ~-~-i:
                theta = arccos(1-gamma)
                qc = QuantumCircuit(3)
                qc.x(1)
                qc.cry(theta, 1, 0)
                self.cor_op.append(Operator(qc))
        pass
    
    def run_SDP(self, gamma, state, _ATOL = 1e-2):
        '''Executes the QEC with SDP
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Optimal fidelity
        '''
        
        # Expand state with ancilla qubits
        from qiskit.quantum_info import DensityMatrix, Choi
        state = DensityMatrix([1] + [0] * 3).expand(state.expand(DensityMatrix([1, 0])))
        noise_ops = damp_err(gamma, self.num_qubit, self.num_error)
        
        # Perform seesaw
#         print('Initiating see-saw...')
        seesaw = QECC_seesaw(Choi(self.enc_op), self.num_qubit, self.num_error)
#         print('N0')
        seesaw.cal_N0(noise_ops)
#         print('N1')
        seesaw.cal_R1(state)
#         print('Runing see-saw...')
        seesaw.run(_ATOL)
        return seesaw.fidelity
