def meas_gen(n, generator):
    '''Creates operator to measure stabilizer generator
    n [int]: Number of physical qubit in one logical qubit
    generator [list]: Stabilizer generator
    '''
    
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    
    # Create coresponding quantum circuit
    qc = QuantumCircuit(n + len(generator))
    qc.h(range(len(generator)))
    
    # Apply controlled gates
    for i in range(len(generator)):
        gen = generator[i]
        for j in range(len(generator[i])):
            if generator[i][j] == 'X':
                qc.cx(i, ~j)
            if generator[i][j] == 'Z':
                qc.cz(i, ~j)
    
    qc.h(range(len(generator)))
    return Operator(qc)

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
		return 0

	from itertools import combinations, product
	from numpy import eye, zeros, sqrt, kron

	_E0 = [eye(2), zeros((2,2))]
	_E0[0][1][1] = (1-gamma)**0.5
	_E0[1][0][1] = gamma**0.5

	E0 = []

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
    '''This method produces noise operators
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
    '''This method produces noise operators
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

# Single-qubit error
# QEC5 = [1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000007, 1.0000000000000007, 1.0000000000000002]
# SDP5 = [1.0000002549183629, 0.9997671640728529, 1.0000002300359045, 0.9999840565349504, 0.9999960672176433, 1.0001129441080925, 0.9999999750827795, 0.9999744824008846, 1.0000048230544263, 0.9999988395145082]
# QEC4 = [1.0000000000000007, 0.9999980262995942, 0.9999611269381936, 0.987698832087293, 0.9825114950565298, 0.9781457518774508, 0.950263532736376, 0.9569585047089221, 0.9553072681454774, 0.95345810363005]
# SDP4 = [1.0000027111704957, 0.9999503676041123, 0.9997955265638233, 0.9995351523267824, 0.9991401074173436, 0.998586588633571, 0.9978772727797605, 0.9970164530489534, 0.9959260304000603, 0.994600306954321]
# SDP3 = [0.9999921350873215, 0.9906089337035794, 0.9809584217572476, 0.9709684567253398, 0.9605967068219103, 0.9499448694929827, 0.9388059037944229, 0.9272073661951072, 0.9151429896106051, 0.9023543616853213]
# sing = [1.0000000000000002, 0.9720237690148861, 0.9436267430132539, 0.9147687979209718, 0.8854029962885429, 0.8554738483549543, 0.8249149571305299, 0.7936457577630639, 0.7615668851388541, 0.7285533905932737]

# All-qubit error
# QEC5 = [1.0000000000000007, 0.9978537181770681, 0.9790429600033382, 0.9773056466496146, 0.9496330369865127, 0.9113986807045045, 0.8229223175688783, 0.8263338754521128, 0.796924984663286, 0.7611875673007802]
# SDP5 = [0.9999994298140122, 0.9970286937957573, 0.9883103821922707, 0.9737696276266068, 0.9536410895719232, 0.9282743457811966, 0.8981346397752208, 0.8638069536899622, 0.8260386259557286, 0.7855512674753571]
# QEC4 = [1.0000000000000007, 0.9948199639201005, 0.9741585280321657, 0.9671517400142405, 0.9365542769870459, 0.88064599819042, 0.8599184935231713, 0.8174868321341705, 0.7731300032247478, 0.7178584913837306]
# SDP4 = [1.0000027111705263, 0.9961432518104206, 0.9846151030111905, 0.9655106171613803, 0.9391284566949196, 0.9059231469180817, 0.8664304089032289, 0.8216944928685231, 0.7729667813552271, 0.7258206000691018]
# SDP3 = [0.9999921350873772, 0.971451357491961, 0.9413788877613614, 0.9097042913355827, 0.8765902617746641, 0.8418248540387269, 0.805589945678399, 0.76775306487214, 0.7283989767325174, 0.6875117554901977]

class QECC_seesaw:
    def __init__(self, enc_op, num_qubit, num_err):
        self.enc_op = enc_op
        self.num_qubit = num_qubit
        self.num_error = min(num_qubit, num_err)
        self.fidelity = 1
        pass
    
    def cal_N0(self, noise_ops):
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
    
    '''def normalize(self, op, gamma):
    	from qiskit.quantum_info import state_fidelity, DensityMatrix, Choi, partial_trace, Operator
    	from numpy import kron, eye, conj, transpose
    	
    	state = DensityMatrix([[0.5, 0], [0, 0.5]])
    	anc = [1] + [0] * ~-2**(~-self.num_qubit)
    	ancilla = DensityMatrix(anc)
    	state = ancilla.expand(state)
    	state = partial_trace(self.enc_op.data@kron(state.partial_transpose(range(self.num_qubit)), eye(2**self.num_qubit)), range(self.num_qubit, self.num_qubit<<1))
#     	print(state)
    	noise_ops = damp_err(gamma, self.num_qubit)
#     	print(state.conjugate(), Operator(conj(noise_ops[0])))
    	st = state.conjugate().evolve(Operator(conj(noise_ops[0])))
#     	st = noise_ops[0]@state.conjugate().data@transpose(conj(noise_ops[0]))
    	dm = partial_trace(self.chk_op[0]@kron(st.partial_transpose(range(self.num_qubit)), eye(2**self.num_qubit)), range(self.num_qubit, self.num_qubit<<1)).evolve(Operator(noise_ops[0]))
#     	dm = noise_ops[0]@partial_trace(self.chk_op[0]@kron(st.partial_transpose(range(self.num_qubit)), eye(2**self.num_qubit)), [1])@transpose(conj(noise_ops[0]))
    	for noise in noise_ops[1:]:
		    st = state.conjugate().evolve(Operator(conj(noise)))
# 		    st = noise@state.conjugate().data@transpose(conj(noise))
		    dm += partial_trace(self.chk_op[0]@kron(st.partial_transpose(range(self.num_qubit)), eye(2**self.num_qubit)), range(self.num_qubit, self.num_qubit<<1)).evolve(Operator(noise))    
    	if op == 'final':
    		dm = partial_trace(self.dec_op[0].data@kron(dm.partial_transpose(range(self.num_qubit)), eye(2**self.num_qubit)), range(self.num_qubit, self.num_qubit<<1))
    	return [self.chk_op[0]/dm.trace()], partial_trace(DensityMatrix(dm/dm.trace()), range(~-self.num_qubit))'''
    
    def run(self, gamma, _ATOL = 1e-2):
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
#             print()
#             break
        self.fidelity = fidelity
#         print()
        print(fidelity)
        pass
        
class five_qubit_code:
    def __init__(self, num_err):
        self.num_qubit = 5
        self.num_error = min(self.num_qubit, num_err)
        self.generator = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']    #Stabilizer generator
        self.L0 = [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]    #Logical 0
        self.L1 = [31, 13, 22, 11, 21, -4, -25, -7, -2, -28, -1, -16, -14, -19, -8, 26]    #Logical 1
        self.enc_op = self._get_enc_op()    #Encoding operator
        self.codebook = {0: 'IIIII', 1: 'IXIII', 2: 'IIIIZ', 3: 'IIXII', 4: 'IIZII', 5: 'ZIIII', 6: 'IIIXI',
                         7: 'IIYII', 8: 'XIIII', 9: 'IIIZI', 10: 'IZIII', 11: 'IYIII', 12: 'IIIIX', 13: 'YIIII',
                         14: 'IIIIY', 15: 'IIIYI'}    #Syndrome measurement: Correction
        self.synd_op = meas_gen(5, self.generator)    #Syndrome measurement operator
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
    
    def run(self, state, gamma):
        '''Executes the QEC
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Corrected state
        '''
        
        from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
        from numpy import transpose
        
        # Encode the state
        anc = [1] + [0] * ~-2**(~-self.num_qubit)
        ancilla = DensityMatrix(anc)
        state = ancilla.expand(state)
        state = state.evolve(self.enc_op)
        
        # Apply noise
        noise_ops = Kraus(damp_err(gamma, self.num_qubit, self.num_error))
        state = state.evolve(noise_ops)
        
        # Syndrome measurement
        anc = [1] + [0] * ~-2**len(self.generator)
        ancilla = DensityMatrix(anc)
        state = ancilla.expand(state)
        state = state.evolve(self.synd_op)
        synd_res, state = state.measure(range(len(self.generator)))
        state = partial_trace(state, range(len(self.generator)))
        
        # Apply correction operation
        self._get_cor_op(synd_res)
        state = state.evolve(self.cor_op)
        
        # Decode the state
        state = state.evolve(self.enc_op.transpose())    #Decoding operation is conjugate transpose of Encoding op.
        return partial_trace(state, range(4))
    
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
        self.synd_op = self._get_synd_op()    #Syndrome measurement operator
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
    
    def _get_synd_op(self):
        '''Creates syndrome measurement operation'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        # Create corresponding quantum circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Apply CNOT gates
        qc.cx([0, 2], [1, 3])
        
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
    
    def run(self, state, gamma):
        '''Executes the QEC
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Corrected state
        '''
        
        from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
        from numpy import transpose
        
        # Encode the state
        state = DensityMatrix([1] + [0] * 3).expand(state.expand(DensityMatrix([1, 0])))
        state = state.evolve(self.enc_op)
        
        # Apply noise
        noise_ops = Kraus(damp_err(gamma, self.num_qubit, self.num_error))
        state = state.evolve(noise_ops)
        
        # Syndrome measurement
        synd_res, state = state.evolve(self.synd_op).measure([1, 3])
        state = partial_trace(state, [1, 3])
        
        # Apply correction operation
        self._get_cor_op(gamma)
        if not int(synd_res, 2):
            state = state.evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([1])[1], [1])
        elif not ~-int(synd_res, 2):
            state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([0])[1], [0, 1])
        elif not ~-~-int(synd_res, 2):
            state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([0])[1], [0, 2])
        
        return state
    
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

class three_qubit_code:
    def __init__(self, num_err):
        self.num_qubit = 3
        self.num_error = min(self.num_qubit, num_err)
        self.enc_op = self._get_enc_op()    #Encoding operator
#         self.synd_op = self._get_synd_op()    #Syndrome measurement operator
        self.cor_op = None    #Correction operator
        pass
    
    def _get_enc_op(self):
        '''Creates Encoding operator'''
        
        from numpy import array, sqrt
        from qiskit.quantum_info import Operator
        
        op = array([[ 0, 0, 0, 1, 0, 0, 0, 0], [ 1/sqrt(3), sqrt(2/3), 0, 0, 0, 0, 0, 0], [ 1/sqrt(3), -1/sqrt(6), 1/sqrt(2), 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 0, 0, 1],
                        [ 1/sqrt(3), -1/sqrt(6), -1/sqrt(2), 0, 0, 0, 0, 0], [ 0, 0, 0, 0, 0, 1, 0, 0], [ 0, 0, 0, 0, 0, 0, 1, 0], [ 0, 0, 0, 0, 1, 0, 0, 0]])
      
        return Operator(op)
    
    def _get_synd_op(self):
        '''Creates syndrome measurement operation'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        # Create corresponding quantum circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Apply CNOT gates
        qc.cx([0, 2], [1, 3])
        
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
    
    def run(self, state, gamma):
        '''Executes the QEC
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Corrected state
        '''
        
        from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
        from numpy import transpose
        
        # Encode the state
        state = DensityMatrix([1] + [0] * 3).expand(state.expand(DensityMatrix([1, 0])))
        state = state.evolve(self.enc_op)
        
        # Apply noise
        noise_ops = Kraus(damp_err(gamma, self.num_qubit, self.num_error))
        state = state.evolve(noise_ops)
        
        # Syndrome measurement
        synd_res, state = state.evolve(self.synd_op).measure([1, 3])
        state = partial_trace(state, [1, 3])
        
        # Apply correction operation
        self._get_cor_op(gamma)
        if not int(synd_res, 2):
            state = state.evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([1])[1], [1])
        elif not ~-int(synd_res, 2):
            state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([0])[1], [0, 1])
        elif not ~-~-int(synd_res, 2):
            state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
            state = partial_trace(state.measure([0])[1], [0, 2])
        
        return state
    
    def run_SDP(self, gamma, state, _ATOL = 1e-2):
        '''Executes the QEC with SDP
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Optimal fidelity
        '''
        
        # Expand state with ancilla qubits
        from qiskit.quantum_info import DensityMatrix, Choi
        state = DensityMatrix([1] + [0] * 3).expand(state)
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
