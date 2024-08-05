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

def damp_err(gamma, n):
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

# def choi(A):
#     '''This method creates Choi representation of given matrix'''
#     from numpy import matrix, concatenate, array, matmul
#     vec = concatenate(array(A), axis = None)
#     V_con = [[v] for v in vec]
#     return matmul(V_con, matrix(vec))

class SDP:
    def __init__(self, num_qubit, gamma, state):
        '''
        num_qubit [int]: Number of physical qubit in one logical qubit
        gamma [float]: Damping probability
        state [dict]: Ensumble of density matrix
        '''
        
        if not isinstance(num_qubit, int) or num_qubit <= 0:
            raise ValueError(f"Number of qubit should be positive integer. Given {num_qubit}")
        
        from numpy import array
        
        self.n = num_qubit
        self.noise = self._get_noise_ops(gamma)
        self.state = choi(state)    #Input state
        self.X = None    #Variable
        self.constraints = None
        self.objective = None
        pass
    
    def _get_noise_ops(self, gamma):
        '''Creates noise operations in Choi representation'''
        noise_ops = damp_err(gamma, self.n)
        return [choi(E) for E in noise_ops]
    
    def solve(self, maximize):
        '''This method solves the SDP
        maximize [bool]: If True solves maximization problem, else solves minimization problem
        Returns: Optimal value
        '''
        
        if not isinstance(maximize, bool):
            raise TypeError('Not an optimization problem.')
        
        if maximize:
            from cvxpy import Problem, Maximize
            return Problem(Maximize(self.objective), self.constraints).solve()
        else:
            from cvxpy import Problem, Maximize
            return Problem(Minimize(self.objective), self.constraints).solve()

class QECC_seesaw:
    def __init__(self, level, enc_op, num_qubit):
        self.level = level
        self.noise_ops = [[] for _ in range(self.level)]
        self.enc_op = enc_op
        self.num_qubit = num_qubit
        self.fidelity = 1
        pass
    
    def cal_noise_ops(self, Gamma):
        from numpy import outer, zeros
        
        if isinstance(Gamma, float):
            Gamma = [Gamma]
        elif len(Gamma) != self.level:
            raise ValueError(f"Number of damping parameters {len(Gamma)} is not same as number of errors {self.level}!")
        noise_ops = []
        for gamma in Gamma:
            noise_ops.append(damp_err(gamma, self.num_qubit))
        if not ~-len(Gamma):
            noise_ops *= self.level
        for l in range(self.level):
            nops = []
            for i in range(2**self.num_qubit):
                nops.append([])
                for j in range(2**self.num_qubit):
                    nops[i].append([])
                    nop = zeros((2**self.num_qubit, 2**self.num_qubit))
                    for noise_op in noise_ops[l]:
                        nop += outer(noise_op.conjugate()[:,i], noise_op[:,j])
                    nops[i][j] = nop
            self.noise_ops[l] = nops
        pass
    
    def cal_state_ops(self, state):
        from numpy import outer
        
        state_ops = []
        for i in range(2**self.num_qubit):
            state_ops.append([])
            for j in range(2**self.num_qubit):
                state_ops[i].append([])
                sop = outer(state.data[:,i], state.data.conjugate()[:,j])
                state_ops[i][j] = sop
        self.state_ops = state_ops
        pass
    
    def run(self, _ATOL = 1e-2):
        from cvxpy import partial_trace, trace, real, Variable, Problem, Maximize
        from numpy import eye, kron, trace as tr, zeros, array
        from os import getpid
        
        itr = 0
        while(True):
            itr += 1
            # SDP for Decoding
            print(f'PID {getpid()}: Running Iteration {itr}...')

            # Define variable
            variable = Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian=True)

            # Define constraints
            constraints = [variable >> 0, partial_trace(variable, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            Obj = array(self.enc_op)[0][0]*kron(self.noise_ops[0][0][0], self.state_ops[0][0])
            for i in range(2**self.num_qubit):
                for j in range(2**self.num_qubit):
                    for k in range(2**self.num_qubit):
                        for l in range(2**self.num_qubit):
                            Obj += array(self.enc_op)[l*2**self.num_qubit+j][k*2**self.num_qubit+i]*kron(self.noise_ops[0][i][j], self.state_ops[k][l])
            Obj -= array(self.enc_op)[0][0]*kron(self.noise_ops[0][0][0], self.state_ops[0][0])
            objective = real(trace(variable@Obj))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.dec_op = variable.value
#             print(self.fidelity, fidelity, end = '\r')
            if abs(fidelity - self.fidelity) < _ATOL:
                break
            self.fidelity = fidelity
                
            # SDP for Encoding
#             print(f'Iteration {itr}: Running SDP for encoding... {fidelity} {self.fidelity}', end = ', ')

            # Define variable
            variable = Variable((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), hermitian=True)

            # Define constraints
            constraints = [variable >> 0, partial_trace(variable, (2**self.num_qubit, 2**self.num_qubit), 1) == eye(2**self.num_qubit)]

            # Define objective function
            mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
            mat[0][0] = 1
            Obj = tr(self.dec_op@kron(self.noise_ops[0][0][0], self.state_ops[0][0]))*mat
            for i in range(2**self.num_qubit):
                for j in range(2**self.num_qubit):
                    for k in range(2**self.num_qubit):
                        for l in range(2**self.num_qubit):
                            mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                            mat[k*2**self.num_qubit+i][l*2**self.num_qubit+j] = 1
                            Obj += tr(self.dec_op@kron(self.noise_ops[0][i][j], self.state_ops[k][l]))*mat
            mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
            mat[0][0] = 1
            Obj -= tr(self.dec_op@kron(self.noise_ops[0][0][0], self.state_ops[0][0]))*mat
            objective = real(trace(variable@Obj))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.enc_op = variable.value
#             print(self.fidelity, fidelity, end = '\r')
            if abs(fidelity - self.fidelity) < _ATOL:
                break
            self.fidelity = fidelity
            print()
        self.fidelity = fidelity
        print()
        pass
        

class four_qubit_code:
    def __init__(self):
        self.num_qubit = 2
#         self.enc_op = self._get_enc_op()    #Encoding operator
#         self.synd_op = self._get_synd_op()    #Syndrome measurement operator
        self.cor_op = None    #Correction operator
        pass
    
    def _get_enc_op(self):
        '''Creates Encoding operator'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        # Create corresponding quantum circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Encode operations
        qc.h(0)
        qc.cx([3, 0, 0, 0], [2, 1, 2, 3])
        
        return Operator(qc)
    
    def _get_synd_op(self):
        '''Creates syndrome measurement operation'''
        
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        # Create corresponding quantum circuit
        qc = QuantumCircuit(self.num_qubit)
        
        # Apply operations
        qc.cx([0, 3], [1, 2])
        
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
        
        from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace, state_fidelity
        
        # Expand state with ancilla qubits
        anc = [1] + [0] * ~-2**(~-self.num_qubit)
        ancilla = DensityMatrix(anc)
        state_exp = ancilla.expand(state)
        
        # Encode the state
        matrix = DensityMatrix(state_exp).evolve(self.enc_op)
        
        # Apply noise
        noise_ops = Kraus(damp_err(gamma, self.num_qubit))
        matrix = matrix.evolve(noise_ops)
        
        # Syndrome measurement
        synd_res, matrix = matrix.evolve(self.synd_op).measure([1, 2])
        matrix = partial_trace(matrix, [1, 2])
        
        # Apply correction operation
        self._get_cor_op(gamma)
        if not int(synd_res, 2):
            matrix = matrix.evolve(self.cor_op[int(synd_res, 2)])
            matrix = partial_trace(matrix.measure([1])[1], [1])
            fid = state_fidelity(DensityMatrix(state), matrix)
        elif not ~-int(synd_res, 2):
            matrix = DensityMatrix([1, 0]).expand(matrix).evolve(self.cor_op[int(synd_res, 2)])
            matrix = partial_trace(matrix.measure([0])[1], [0, 1])
            fid = state_fidelity(DensityMatrix(state), matrix)
        elif not ~-~-int(synd_res, 2):
            matrix = DensityMatrix([1, 0]).expand(matrix).evolve(self.cor_op[int(synd_res, 2)])
            matrix = partial_trace(matrix.measure([0])[1], [0, 2])
            fid = state_fidelity(DensityMatrix(state), matrix)
        else:
            fid = 0
        
        return fid
    
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
        
        # Perform seesaw
        seesaw = QECC_seesaw(1, Choi(self._init_enc(self.num_qubit)), self.num_qubit)
        seesaw.cal_noise_ops(gamma)
        seesaw.cal_state_ops(state)
        seesaw.run(_ATOL)
        return (seesaw.fidelity, seesaw.enc_op, seesaw.dec_op)
    
    @staticmethod
    def _init_enc(num_qubits):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
#         qc = QuantumCircuit(num_qubits)
#         qc.h(0)
#         qc.cx([3, 0, 0, 0], [2, 1, 2, 3])
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return Operator(qc)
        
#         from cvxpy import partial_trace, trace, real, Variable, kron as tens
#         from numpy import eye, kron, transpose, matrix, sqrt
        
#         enc_op = choi(self.enc_op)
#         chk_op = choi(self.synd_op)
#         itr = 1
#         while(True):
#             # Construct SDP for DECODING
#             print('Running SDP for decoding')
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.D = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.D >> 0, partial_trace(prob.D, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = []
#             for noise0 in prob.noise:
#                 for noise1 in prob.noise:
#                     Obj.append(kron(kron(noise0, noise1), choi(eye(2**(prob.n<<1))))@tens(prob.D, kron(chk_op, enc_op))@kron(prob.state, choi(eye(2**(prob.n<<2)))))
#             obj = Obj[0]
#             for o in Obj[1:]:
#                 obj += o
#             prob.objective = real(trace(obj))
            
#             fidelity = prob.solve(True)
#             dec_op = prob.D.value
#             print(itr, fidelity, end = ', ')
#             if 1 - fidelity < ATOL:
#                 break
            
#             # Construct SDP for ENCODING
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.E = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.E >> 0, partial_trace(prob.E, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = []
#             for noise0 in prob.noise_ops:
#                 for noise1 in prob.noise_ops:
#                     Obj.append(kron(kron(noise0, noise1), choi(eye(2**(prob.n<<1))))@tens(kron(dec_op, chk_op), prob.E)@kron(prob.state, choi(eye(2**(prob.n<<2)))))
#             obj = Obj[0]
#             for o in Obj[1:]:
#                 obj += o
#             prob.objective = real(trace(obj))
            
#             fidelity = prob.solve(True)
#             enc_op = prob.E.value
#             print(fidelity, end = ', ')
#             if 1 - fidelity < ATOL:
#                 break
            
#             # Construct SDP for CHECKING
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.C = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.C >> 0, partial_trace(prob.C, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = []
#             for noise0 in prob.noise_ops:
#                 for noise1 in prob.noise_ops:
#                     Obj.append(kron(kron(noise0, noise1), choi(eye(2**(prob.n<<1))))@tens(tens(dec_op, prob.C), enc_op)@kron(prob.state, choi(eye(2**(prob.n<<2)))))
#             obj = Obj[0]
#             for o in Obj[1:]:
#                 obj += o
#             prob.objective = real(trace(obj))
            
#             fidelity = prob.solve(True)
#             chk_op = prob.C.value
#             print(fidelity)
#             itr = -~itr
#             if 1 - fidelity < ATOL:
#                 break
#         return (fidelity, enc_op, chk_op, dec_op)
        
class five_qubit_code:
    def __init__(self):
        self.num_qubit = 5
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
        
        from numpy import zeros, transpose, sqrt
        from qiskit.quantum_info import Operator
        
        # Operator is initially zero
        enc_op = zeros((2, 2**self.num_qubit))
        
        # Put 1 and -1 depending on L0 and L1
        # if i in Lj, (j, abs(i))-th element of the operator will be 1 if i is no-negative and -1 otherwise
        for i in self.L0:
            v = 1/sqrt(len(self.L0))
            if i != abs(i):
                v = -v
            enc_op[0][abs(i)] = v
        for i in self.L1:
            v = 1/sqrt(len(self.L1))
            if i != abs(i):
                v = -v
            enc_op[1][abs(i)] = v
        return Operator(transpose(enc_op))
    
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
        state = DensityMatrix(state).evolve(self.enc_op)
        
        # Apply noise
        noise_ops = Kraus(damp_err(gamma, self.num_qubit))
        state = state.evolve(noise_ops)
        
        # Syndrome measurement
        anc = [1] + [0] * ~-(2**len(self.generator))
        state = DensityMatrix(anc).expand(state).evolve(self.synd_op)
        synd_res, state = state.measure(range(len(self.generator)))
        state = partial_trace(state, range(len(self.generator)))
        
        # Apply correction operation
        self._get_cor_op(synd_res)
        state = state.evolve(self.cor_op)
        
        # Decode the state
        state = state.evolve(self.enc_op.transpose())    #Decoding operation is conjugate transpose of Encoding op.
        return state
    
    def run_SDP(self, gamma, state):
        '''Executes the QEC with SDP
        state [list or numpy array]: Density matrix
        gamma [float]: Damping probability
        Returns: Optimal fidelity
        '''
        
        from cvxpy import partial_trace, trace, real, Variable
        from numpy import eye, kron, transpose, matrix, sqrt
        
        # Construct SDP
        prob = SDP(self.num_qubit, gamma, state)
        
        # Define variable
        prob.X = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)
        
        # Define constraints
        prob.constraints = [prob.X >> 0, partial_trace(prob.X, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]
        
        # Define objective function
        Cv = []
        for st, pr in prob.state.items():
            s_v = [0, 0]
            s_v[st] = 1
            anc = [1] + [0] * ~-(2**~-prob.n)
            sv = [[i] for i in kron(s_v, anc)]
            for noise in prob.noise_ops:
                Cv.append(sqrt(pr)*vec(sv@matrix(s_v)@self.enc_op.transpose().data@transpose(noise)))
        C = Cv[0].getH().dot(Cv[0])
        for c in Cv[1:]:
            C += c.getH().dot(c)
        prob.objective = real(trace(prob.X@C))
        
        return prob.solve(True)

# '''def meas_gen(n, generator):
#     '''Creates operator to measure stabilizer generator
#     n [int]: Number of physical qubit in one logical qubit
#     generator [list]: Stabilizer generator
#     '''
    
#     from qiskit import QuantumCircuit
#     from qiskit.quantum_info import Operator
    
#     # Create coresponding quantum circuit
#     qc = QuantumCircuit(n + len(generator))
#     qc.h(range(len(generator)))
    
#     # Apply controlled gates
#     for i in range(len(generator)):
#         gen = generator[i]
#         for j in range(len(generator[i])):
#             if generator[i][j] == 'X':
#                 qc.cx(i, ~j)
#             if generator[i][j] == 'Z':
#                 qc.cz(i, ~j)
    
#     qc.h(range(len(generator)))
#     return Operator(qc)

# def damp_err(gamma, n):
#     '''This method produces noise operators
#     gamma [float]: Damping probability
#     n [int]: Number of qubit
#     '''
    
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError(f"Number of qubit should be positive integer. Given {n}")
#     if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
#         raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
#     from numpy import eye, zeros, kron, sqrt
    
#     _E = [eye(2), zeros((2, 2))]
#     _E[0][1][1] = sqrt(1-gamma)
#     _E[1][0][1] = sqrt(gamma)
    
#     E = _E.copy()
#     for m in range(1, n):
#         E_ = []
#         for i in _E:
#             for j in E:
#                 E_.append(kron(i, j))
#         E = E_.copy()
#     return E

# def choi(A):
#     '''This method creates Choi representation of given matrix'''
#     from numpy import matrix, concatenate, array, matmul
#     vec = concatenate(array(A), axis = None)
#     V_con = [[v] for v in vec]
#     return matmul(V_con, matrix(vec))

# class SDP:
#     def __init__(self, num_qubit, gamma, state):
#         '''
#         num_qubit [int]: Number of physical qubit in one logical qubit
#         gamma [float]: Damping probability
#         state [<DensityMatrix>]: Density matrix
#         '''
        
#         if not isinstance(num_qubit, int) or num_qubit <= 0:
#             raise ValueError(f"Number of qubit should be positive integer. Given {num_qubit}")
        
#         from numpy import array
        
#         self.n = num_qubit
#         self.noise = self._get_noise_ops(gamma)
#         self.state = choi(state)    #Input state
#         self.X = None    #Variable
#         self.constraints = None
#         self.objective = None
#         pass
    
#     def _get_noise_ops(self, gamma):
#         '''Creates noise operations in Choi representation'''
#         noise_ops = damp_err(gamma, self.n)
#         return [choi(E) for E in noise_ops]
    
#     def solve(self, maximize):
#         '''This method solves the SDP
#         maximize [bool]: If True solves maximization problem, else solves minimization problem
#         Returns: Optimal value
#         '''
        
#         if not isinstance(maximize, bool):
#             raise TypeError('Not an optimization problem.')
        
#         if maximize:
#             from cvxpy import Problem, Maximize
#             return Problem(Maximize(self.objective), self.constraints).solve()
#         else:
#             from cvxpy import Problem, Maximize
#             return Problem(Minimize(self.objective), self.constraints).solve()

# class four_qubit_code:
#     def __init__(self):
#         self.num_qubit = 4
#         self.enc_op = self._get_enc_op()    #Encoding operator
#         self.synd_op = self._get_synd_op()    #Syndrome measurement operator
#         self.cor_op = None    #Correction operator
#         pass
    
#     def _get_enc_op(self):
#         '''Creates Encoding operator'''
        
#         from qiskit import QuantumCircuit
#         from qiskit.quantum_info import Operator
        
#         # Create corresponding quantum circuit
#         qc = QuantumCircuit(self.num_qubit)
        
#         # Encode operations
#         qc.h(0)
#         qc.cx([3, 0, 0, 0], [2, 1, 2, 3])
        
#         return Operator(qc)
    
#     def _get_synd_op(self):
#         '''Creates syndrome measurement operation'''
        
#         from qiskit import QuantumCircuit
#         from qiskit.quantum_info import Operator
        
#         # Create corresponding quantum circuit
#         qc = QuantumCircuit(self.num_qubit)
        
#         # Apply operations
#         qc.cx([0, 3], [1, 2])
        
#         return Operator(qc)
    
#     def _get_cor_op(self, gamma):
#         '''Creates correction operation
#         gamma [float]: Damping probability
#         '''
        
#         from numpy import arctan, arccos, pi
#         from qiskit import QuantumCircuit
#         from qiskit.quantum_info import Operator
        
#         self.cor_op = []
#         for i in range(3):
#             if not i:
#                 theta = arctan((1-gamma)**2)
#                 qc = QuantumCircuit(2)
#                 qc.cx(1, 0)
#                 qc.ry(2*theta, 1)
#                 qc.cry(pi/2-2*theta, 0, 1)
#                 self.cor_op.append(Operator(qc))
#             elif not ~-i:
#                 theta = arccos(1-gamma)
#                 qc = QuantumCircuit(3)
#                 qc.x(2)
#                 qc.cry(theta, 2, 0)
#                 self.cor_op.append(Operator(qc))
#             elif not ~-~-i:
#                 theta = arccos(1-gamma)
#                 qc = QuantumCircuit(3)
#                 qc.x(1)
#                 qc.cry(theta, 1, 0)
#                 self.cor_op.append(Operator(qc))
#         pass
    
#     def run(self, state, gamma):
#         '''Executes the QEC
#         state [list or numpy array]: Density matrix
#         gamma [float]: Damping probability
#         Returns: Corrected state
#         '''
        
#         from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
        
#         # Encode the state
#         state = DensityMatrix(state).evolve(self.enc_op)
        
#         # Apply noise
#         noise_ops = Kraus(damp_err(gamma, self.num_qubit))
#         state = state.evolve(noise_ops)
        
#         # Syndrome measurement
#         synd_res, state = state.evolve(self.synd_op).measure([1, 2])
#         state = partial_trace(state, [1, 2])
        
#         # Apply correction operation
#         self._get_cor_op(gamma)
#         if not int(synd_res, 2):
#             state = state.evolve(self.cor_op[int(synd_res, 2)])
#             state = partial_trace(state.measure([1])[1], [1])
#         elif not ~-int(synd_res, 2):
#             state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
#             state = partial_trace(state.measure([0])[1], [0, 1])
#         elif not ~-~-int(synd_res, 2):
#             state = DensityMatrix([1, 0]).expand(state).evolve(self.cor_op[int(synd_res, 2)])
#             state = partial_trace(state.measure([0])[1], [0, 2])
        
#         return state
    
#     def run_SDP(self, gamma, state, ATOL = 1e-2):
#         '''Executes the QEC with SDP
#         state [list or numpy array]: Density matrix
#         gamma [float]: Damping probability
#         Returns: Optimal fidelity
#         '''
        
#         from cvxpy import partial_trace, trace, real, Variable, kron as tens
#         from numpy import eye, kron, transpose, matrix, sqrt, trace as tr
#         from qiskit.quantum_info import DensityMatrix
        
#         for i in range(~-self.num_qubit):
#             state = DensityMatrix([1, 0]).expand(state)
        
#         enc_op = choi(self.enc_op)
#         chk_op = choi(self.synd_op)
#         itr = 1
#         while(True):
#             # Construct SDP for DECODING
#             print('Running SDP for decoding')
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.D = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.D >> 0, partial_trace(prob.D, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = []
#             for noise in prob.noise:
#                 Obj.append(noise@prob.D@prob.state)
#             obj_D = Obj[0]
#             for o in Obj[1:]:
#                 obj_D += o
#             Obj = []
#             for noise in prob.noise:
#                 Obj.append(noise@chk_op)
#             obj_C = Obj[0]
#             for o in Obj[1:]:
#                 obj_C += o
#             obj_E = enc_op
#             prob.objective = real(trace(obj_D)*trace(obj_C)*trace(obj_E))
#             print(tr(obj_C), tr(obj_E))
#             fidelity = prob.solve(True)
#             dec_op = prob.D.value
#             print(itr, fidelity, end = ', ')
#             if 1 - fidelity < ATOL:
#                 break
            
#             # Construct SDP for ENCODING
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.E = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.E >> 0, partial_trace(prob.E, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = prob.E
#             prob.objective = real(trace(obj))
            
#             fidelity = prob.solve(True)
#             enc_op = prob.E.value
#             print(fidelity, end = ', ')
#             if 1 - fidelity < ATOL:
#                 break
            
#             # Construct SDP for CHECKING
#             prob = SDP(self.num_qubit, gamma, state)

#             # Define variable
#             prob.C = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)

#             # Define constraints
#             prob.constraints = [prob.C >> 0, partial_trace(prob.C, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]

#             # Define objective function
#             Obj = []
#             for noise in prob.noise:
#                 Obj.append(noise@prob.C)
#             obj = Obj[0]
#             for o in Obj[1:]:
#                 obj += o
#             prob.objective = real(trace(obj))
            
#             fidelity = prob.solve(True)
#             chk_op = prob.C.value
#             print(fidelity)
#             itr = -~itr
#             if 1 - fidelity < ATOL:
#                 break
#         return (fidelity, enc_op, chk_op, dec_op)
        
# class five_qubit_code:
#     def __init__(self):
#         self.num_qubit = 5
#         self.generator = ['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ']    #Stabilizer generator
#         self.L0 = [0, 18, 9, 20, 10, -27, -6, -24, -29, -3, -30, -15, -17, -12, -23, 5]    #Logical 0
#         self.L1 = [31, 13, 22, 11, 21, -4, -25, -7, -2, -28, -1, -16, -14, -19, -8, 26]    #Logical 1
#         self.enc_op = self._get_enc_op()    #Encoding operator
#         self.codebook = {0: 'IIIII', 1: 'IXIII', 2: 'IIIIZ', 3: 'IIXII', 4: 'IIZII', 5: 'ZIIII', 6: 'IIIXI',
#                          7: 'IIYII', 8: 'XIIII', 9: 'IIIZI', 10: 'IZIII', 11: 'IYIII', 12: 'IIIIX', 13: 'YIIII',
#                          14: 'IIIIY', 15: 'IIIYI'}    #Syndrome measurement: Correction
#         self.synd_op = meas_gen(5, self.generator)    #Syndrome measurement operator
#         self.cor_op = None    #Correction operator
#         pass
    
#     def _get_enc_op(self):
#         '''Creates Encoding operator'''
        
#         from numpy import zeros, transpose, sqrt
#         from qiskit.quantum_info import Operator
        
#         # Operator is initially zero
#         enc_op = zeros((2, 2**self.num_qubit))
        
#         # Put 1 and -1 depending on L0 and L1
#         # if i in Lj, (j, abs(i))-th element of the operator will be 1 if i is no-negative and -1 otherwise
#         for i in self.L0:
#             v = 1/sqrt(len(self.L0))
#             if i != abs(i):
#                 v = -v
#             enc_op[0][abs(i)] = v
#         for i in self.L1:
#             v = 1/sqrt(len(self.L1))
#             if i != abs(i):
#                 v = -v
#             enc_op[1][abs(i)] = v
#         return Operator(transpose(enc_op))
    
#     def _get_cor_op(self, synd_res):
#         '''Creates correction operation
#         synd_res [str]: Syndrome measurement result
#         '''
        
#         from qiskit.quantum_info import Operator, Pauli
#         self.cor_op = Operator(Pauli(self.codebook[int(synd_res, 2)]))
#         pass
    
#     def run(self, state, gamma):
#         '''Executes the QEC
#         state [list or numpy array]: Density matrix
#         gamma [float]: Damping probability
#         Returns: Corrected state
#         '''
        
#         from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace
#         from numpy import transpose
        
#         # Encode the state
#         state = DensityMatrix(state).evolve(self.enc_op)
        
#         # Apply noise
#         noise_ops = Kraus(damp_err(gamma, self.num_qubit))
#         state = state.evolve(noise_ops)
        
#         # Syndrome measurement
#         anc = [1] + [0] * ~-(2**len(self.generator))
#         state = DensityMatrix(anc).expand(state).evolve(self.synd_op)
#         synd_res, state = state.measure(range(len(self.generator)))
#         state = partial_trace(state, range(len(self.generator)))
        
#         # Apply correction operation
#         self._get_cor_op(synd_res)
#         state = state.evolve(self.cor_op)
        
#         # Decode the state
#         state = state.evolve(self.enc_op.transpose())    #Decoding operation is conjugate transpose of Encoding op.
#         return state
    
#     def run_SDP(self, gamma, state):
#         '''Executes the QEC with SDP
#         state [list or numpy array]: Density matrix
#         gamma [float]: Damping probability
#         Returns: Optimal fidelity
#         '''
        
#         from cvxpy import partial_trace, trace, real, Variable
#         from numpy import eye, kron, transpose, matrix, sqrt
        
#         # Construct SDP
#         prob = SDP(self.num_qubit, gamma, state)
        
#         # Define variable
#         prob.X = Variable((2**(prob.n<<1), 2**(prob.n<<1)), hermitian=True)
        
#         # Define constraints
#         prob.constraints = [prob.X >> 0, partial_trace(prob.X, (2**prob.n, 2**prob.n), 0) == eye(2**prob.n)]
        
#         # Define objective function
#         Cv = []
#         for st, pr in prob.state.items():
#             s_v = [0, 0]
#             s_v[st] = 1
#             anc = [1] + [0] * ~-(2**~-prob.n)
#             sv = [[i] for i in kron(s_v, anc)]
#             for noise in prob.noise_ops:
#                 Cv.append(sqrt(pr)*vec(sv@matrix(s_v)@self.enc_op.transpose().data@transpose(noise)))
#         C = Cv[0].getH().dot(Cv[0])
#         for c in Cv[1:]:
#             C += c.getH().dot(c)
#         prob.objective = real(trace(prob.X@C))
        
#         return prob.solve(True)
# '''
