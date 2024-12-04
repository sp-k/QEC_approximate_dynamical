cor = True

def damp_err_old(gamma, n):
    '''This method produces noise operators
    gamma [float]: Damping probability
    n [int]: Number of qubit
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, kron, sqrt
    
    _E = [eye(2), eye(2)]
    _E[0][0][0] = 0.5**0.25
    _E[0][1][1] = (0.5*(1-gamma))**0.25
    _E[0][0][1] = sqrt(2*gamma)/(1+(1-gamma)**0.25)
    _E[1][0][0] = 0.5**0.25
    _E[1][1][1] = (0.5*(1-gamma))**0.25
    _E[1][0][1] = -sqrt(2*gamma)/(1+(1-gamma)**0.25)
    
    E = _E.copy()
    for m in range(1, n):
        E_ = []
        for i in _E:
            for j in E:
                E_.append(kron(i, j))
        E = E_.copy()
    return E

def damp_err(gamma, n):
    '''This method produces noise operators
    gamma [float]: Damping probability
    n [int]: Number of qubit
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, kron, sqrt
    
    _E = [eye(2), eye(2)]
    _E[0][0][0] = 0.5**0.25
    _E[0][1][1] = (0.5*(1-gamma))**0.25
    _E[0][0][1] = sqrt(2*gamma)/(1+(1-gamma)**0.25)
    _E[1][0][0] = 0.5**0.25
    _E[1][1][1] = (0.5*(1-gamma))**0.25
    _E[1][0][1] = -sqrt(2*gamma)/(1+(1-gamma)**0.25)
    
    E = []
    for j in range(n):
        E.append(kron(eye(2**j), kron(_E[0]/sqrt(n), eye(2**(n-j-1)))))
        E.append(kron(eye(2**j), kron(_E[1]/sqrt(n), eye(2**(n-j-1)))))
    return E

class QECC_seesaw:
    def __init__(self, enc_op, chk_op, num_qubit):
        self.enc_op = enc_op
        self.chk_op = chk_op
        self.num_qubit = num_qubit
        self.fidelity = 10
        pass
    
    def cal_N0(self, noise_ops):
        from numpy import outer, zeros, kron
        
        nops = []
        for a in range(2**self.num_qubit):
            nops.append([])
            for b in range(2**self.num_qubit):
                nops[a].append([])
                for noise_op in noise_ops:
                    nops[a][b].append(outer(noise_op.conjugate()[:,a], noise_op[:,b]))
        self.N0 = nops
        pass
    
    def cal_R1(self, state):
        from numpy import outer, kron
        
        nops = []
        for s in range(2**self.num_qubit):
            nops.append([])
            for t in range(2**self.num_qubit):
                nops[s].append([])
                nops[s][t] = outer(state.data[:,s], state.data.conjugate()[:,t])
        self.R1 = nops
#         print(len(nops), len(nops[0]), nops)
        pass
    
    @staticmethod
    def get_cd(num_qubits, c, d):
        from numpy import zeros
        
        cd = zeros((2**num_qubits, 2**num_qubits))
        cd[c][d] = 1
        return cd
    
    def obj_dec(self, lock, chk, a, obj):
        from numpy import zeros, array, trace, kron, complex128
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2*self.num_qubit):
                                for j in range(2*self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    tra = 0
                                    for x in range(2**self.num_qubit):
                                        for y in range(2**self.num_qubit):
                                            tra += array(chk)[x*2**self.num_qubit+d][y*2**self.num_qubit+c]*self.N0[a][b][i][y][x]
                                    objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*tra*kron(self.N0[c][d][j], self.R1[s][t])
            print(f'[Dec] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()
    
    '''def obj_dec(self, lock, chk, a, obj):
        from numpy import zeros, array, trace, kron, complex128
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2**self.num_qubit):
                                for j in range(2**self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*trace(array(chk)@kron(self.N0[a][b][i], self.get_cd(self.num_qubit, c, d)))*kron(self.N0[c][d][j], self.R1[s][t])
            print(f'[Dec] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()'''
    
    def obj_chk(self, lock, dec, a, obj):
        from numpy import zeros, array, trace, kron, complex128
        from itertools import product
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2*self.num_qubit):
                                for j in range(2*self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*trace(array(dec)@kron(self.N0[c][d][i], self.R1[s][t]))*kron(self.N0[a][b][j], self.get_cd(self.num_qubit, c, d))
            print(f'[Chk] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()
    
    '''def obj_chk(self, lock, dec, a, obj):
        from numpy import zeros, array, trace, kron, complex128
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2**self.num_qubit):
                                for j in range(2**self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    objL += array(self.enc_op)[t*2**self.num_qubit+b][s*2**self.num_qubit+a]*trace(array(dec)@kron(self.N0[c][d][i], self.R1[s][t]))*kron(self.N0[a][b][j], self.get_cd(self.num_qubit, c, d))
            print(f'[Chk] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()'''
    
    def obj_enc(self, lock, a, obj):
        from numpy import zeros, kron, trace, kron, complex128
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2*self.num_qubit):
                                for j in range(2*self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    for m in range(len(self.chk_op)):
                                        mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                                        mat[s*2**self.num_qubit+a][t*2**self.num_qubit+b] = 1
                                        tra = 0
                                        for x in range(2**self.num_qubit):
                                            for y in range(2**self.num_qubit):
                                                tra += self.chk_op[m][x*2**self.num_qubit+d][y*2**self.num_qubit+c]*self.N0[a][b][i][y][x]
                                        objL += tra*trace(self.dec_op[m]@kron(self.N0[c][d][j], self.R1[s][t]))*mat
            print(f'[Enc] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()
    
    '''def obj_enc(self, lock, a, obj):
        from numpy import zeros, kron, trace, kron, complex128
        from os import getpid
        from time import time
        print(f'{a}: {getpid()}')
        objL = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
        for b in range(2**self.num_qubit):
            start = time()
            for c in range(2**self.num_qubit):
                for d in range(2**self.num_qubit):
                    for s in [0, 2**(~-self.num_qubit)]:
                        for t in [0, 2**(~-self.num_qubit)]:
                            for i in range(2**self.num_qubit):
                                for j in range(2**self.num_qubit):
                                    if cor and i != j:
                                        continue
                                    for m in range(len(self.chk_op)):
                                        mat = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                                        mat[s*2**self.num_qubit+a][t*2**self.num_qubit+b] = 1
                                        objL += trace(self.chk_op[m]@kron(self.N0[a][b][i], self.get_cd(self.num_qubit, c, d)))*trace(self.dec_op[m]@kron(self.N0[c][d][j], self.R1[s][t]))*mat
            print(f'[Enc] Time taken by {a} for b = {b}: {time()-start}')
        lock.acquire()
        obj += objL
        lock.release()'''
    
    def normalize(self, op, gamma):
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
    	return [self.chk_op[0]/dm.trace()], partial_trace(DensityMatrix(dm/dm.trace()), range(~-self.num_qubit))

# 1 = [1.0, 0.9996262428224001, 0.9985438076535248, 0.9967919309209514, 0.9943869615448672, 0.9913238718337404, 0.987575892644087, 0.9830921765790536, 0.9777929315234379, 0.9715609007371916]
# 2 = 
# 3 = 
# 4 = 

## Operation in Overleaf
# 1 = 
# 2 = 
# 3 = 
# 4 = 

## 3-tensor
# 1 = 
# op= 

## single-qubit error
# 2 = [1.0, 0.9996003906700122, 0.99856813249685, 0.9976198963134529, 0.997506252213968, 0.9984262729386069, 0.9979349795069143, 0.996416089989958, 0.9946512657107642, 0.9934770660231799]
# 3 = [1.0, 0.999942632764052, 0.9998037449996635, 0.9998170346254754, 0.9993809691364655, 0.9987318432994146, 0.9981352335291154, 0.9974823248167923, 0.9968043234429083, 0.9961035947958248]
# 4 = 
    
    def run(self, gamma, _ATOL = 1e-2):
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
                d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                shm = shared_memory.SharedMemory(create = True, size = d_size, name = 'object')
                obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)
#                 obj = empty((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
                obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
                jobs = []
                lock = Lock()
                for a in range(2**self.num_qubit):
                    job = Process(target = self.obj_dec, args = (lock, chk, a, obj))
                    job.start()
                    jobs.append(job)
#                     exit()
                for job in jobs:
                    job.join()
                Obj.append(obj.copy())
                shm.close()
                shm.unlink()
            objective = variable[0]@Obj[0]
            for var, obj in zip(variable[1:], Obj[1:]):
                objective += var@obj
            objective = real(trace(objective))
            
            # Solve SDP
#             print('Solving SDP')
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.dec_op = [var.value for var in variable]
            print(self.fidelity, fidelity)
            if abs(fidelity - self.fidelity) < _ATOL:
                break
            self.fidelity = fidelity
            
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
                d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
                shm = shared_memory.SharedMemory(create = True, size = d_size, name = 'object')
                obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)
#                 obj = empty((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
                obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
                jobs = []
                lock = Lock()
                for a in range(2**self.num_qubit):
                    job = Process(target = self.obj_chk, args = (lock, dec, a, obj))
                    job.start()
                    jobs.append(job)
                for job in jobs:
                    job.join()
                Obj.append(obj.copy())
                shm.close()
                shm.unlink()
            objective = variable[0]@Obj[0]
            for var, obj in zip(variable[1:], Obj[1:]):
                objective += var@obj
            objective = real(trace(objective))
            
            # Solve SDP
            fidelity = Problem(Maximize(objective), constraints).solve()
            self.chk_op = [var.value for var in variable]
            print(self.fidelity, fidelity)
#             self.chk_op, dm_final = self.normalize('check', gamma)
#             fidelity = state_fidelity(DensityMatrix([[0.5, 0], [0, 0.5]]), dm_final, False)
#             print(fidelity)
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
            d_size = dtype(complex128).itemsize * prod((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)))
            shm = shared_memory.SharedMemory(create = True, size = d_size, name = 'object')
            Obj = ndarray(shape = (2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), dtype = complex128, buffer = shm.buf)
#             Obj = empty((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
            Obj[:] = zeros((2**(self.num_qubit<<1), 2**(self.num_qubit<<1)), complex128)
            jobs = []
            lock = Lock()
            for a in range(2**self.num_qubit):
                job = Process(target = self.obj_enc, args = (lock, a, Obj))
                job.start()
                jobs.append(job)
            for job in jobs:
                job.join()
            objective = real(trace(variable@Obj.copy()))
            shm.close()
            shm.unlink()
            
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
        _, dm_final = self.normalize('final', gamma)
        fidelity = state_fidelity(DensityMatrix([[0.5, 0], [0, 0.5]]), dm_final, False)
        self.fidelity = fidelity
        print(fidelity)
        pass
        

class four_qubit_code:
    def __init__(self):
        self.num_qubit = 4
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
        noise_ops = damp_err(gamma, self.num_qubit)
        
        # Perform seesaw
#         print('Initiating see-saw...')
        seesaw = QECC_seesaw(Choi(self._init_enc(self.num_qubit)), [Choi(chk) for chk in self._init_chk(self.num_qubit)], self.num_qubit)
#         print('N0')
        seesaw.cal_N0(noise_ops)
#         print('N1')
        seesaw.cal_R1(state)
#         print('Runing see-saw...')
        seesaw.run(_ATOL)
        return (seesaw.fidelity, seesaw.enc_op, seesaw.chk_op, seesaw.dec_op)
    
    @staticmethod
    def _init_enc(num_qubits):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from numpy import eye
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx([3, 0, 0, 0], [2, 1, 2, 3])
#         qc = QuantumCircuit(2)
#         qc.h(0)
#         qc.cx(0, 1)
        
        return Operator(qc)
    
    @staticmethod
    def _init_chk(num_qubits):
        from numpy import eye, kron, zeros
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        
        chk = []
        qc = QuantumCircuit(num_qubits)
        qc.cx([0, 3], [1, 2])
        
        zero = zeros((4, 4))
        M0 = zero.copy()
        M0[0][0] = 1
        M1 = zero.copy()
        M1[1][1] = 1
        M2 = zero.copy()
        M2[2][2] = 1
        chk.append(Operator(qc).compose(kron(eye(2), kron(M0, eye(2)))))
        chk.append(Operator(qc).compose(kron(eye(2), kron(M1, eye(2)))))
        chk.append(Operator(qc).compose(kron(eye(2), kron(M2, eye(2)))))
        return chk
    
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
    
        zero = zeros((2, 2))
        M0 = zero.copy()
        M0[0][0] = 1
        M1 = zero.copy()
        M1[1][1] = 1
        chk.append(Operator(qc).compose(kron(eye(2), M0)))
        chk.append(Operator(qc).compose(kron(eye(2), M1)))
        return chk#[Operator(eye(2))]#chk#[Operator(eye(4))]
    
class n_qubit_code:
    def __init__(self, n, m):
        self.num_qubit = n
        self.num_mem = m
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
        noise_ops = damp_err(gamma, self.num_qubit)
        
        # Perform seesaw
#         print('Initiating see-saw...')
        seesaw = QECC_seesaw(Choi(self._init_enc(self.num_qubit)), [Choi(chk) for chk in self._init_chk(self.num_qubit, self.num_mem)], self.num_qubit)
#         print('N0')
        seesaw.cal_N0(noise_ops)
#         print('N1')
        seesaw.cal_R1(state)
#         print('Runing see-saw...')
        seesaw.run(gamma, _ATOL)
        return (seesaw.fidelity, seesaw.enc_op, seesaw.chk_op, seesaw.dec_op)
    
    @staticmethod
    def _init_enc(num_qubits):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from numpy import eye
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
        return Operator(qc)
    
    @staticmethod
    def _init_chk(num_qubits, num_mem):
        from numpy import eye, kron, zeros, sqrt
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator, Pauli, Choi
        
        chk = []
        qc = QuantumCircuit(num_qubits)
        for i in range(1, num_qubits):
            qc.cx(0, i)
            
        match (num_qubits, num_mem):
            case (1, 1):
                return [Operator(qc).compose(Pauli('X'))]
            case (1, 2):
                return [Operator(qc).compose(Pauli('I'))/sqrt(2), Operator(qc).compose(Pauli('I'))/sqrt(2)]
            case (1, 3):
                return [Operator(qc).compose(Pauli('X'))/3, Operator(qc).compose(Pauli('X'))/3, Operator(qc).compose(Pauli('X'))/3]
            case (2, 1):
                qc.h(0)
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
