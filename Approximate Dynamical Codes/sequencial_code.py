from qiskit.quantum_info import Kraus, DensityMatrix, partial_trace, state_fidelity, Operator, Choi
from numpy import matmul, sqrt, arctan, arccos, pi, outer, eye, kron, trace as tr, zeros, array
from cvxpy import partial_trace as part_tr, trace, real, Variable, Problem, Maximize
from matplotlib.pyplot import subplots, legend, savefig, show
from qiskit import QuantumCircuit

NUM = 1000
num_qubit = 4
num_params = 10
_ATOL = 1e-3
damp_params = [i/(2*~-num_params) for i in range(num_params)]
state = DensityMatrix([[0.5, 0], [0, 0.5]])

fid_QEC4 = [0 for _ in range(len(damp_params))]
fid_SDP4 = [0 for _ in range(len(damp_params))]
fid_sing = [0 for _ in range(len(damp_params))]

def damp_err(gamma, n):
    '''This method produces noise operators
    gamma [float]: Damping probability
    n [int]: Number of qubit
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
    
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

for param in range(len(damp_params)):
    itr = NUM
    gamma = damp_params[param]

    # Expand state with ancilla qubits
    anc = [1] + [0] * ~-2**(~-num_qubit)
    ancilla = DensityMatrix(anc)
    state_exp = ancilla.expand(state)

    # Create quantum circuit for encoding operation
    qc = QuantumCircuit(num_qubit)
    
    # Encode operations
    qc.h(0)
    qc.cx([3, 0, 0, 0], [2, 1, 2, 3])
    
    enc_op = Operator(qc)

    # Actual [[4,1,2]] code
    while(itr):
        try:
            # Encode the state
            matrix = DensityMatrix(state_exp).evolve(enc_op)
            
            # Apply noise
            noise_ops = Kraus(damp_err(gamma, num_qubit))
            matrix = matrix.evolve(noise_ops)
            
            # Create quantum circuit for syndrome measurement
            qc = QuantumCircuit(num_qubit)
            
            # Apply operations
            qc.cx([0, 3], [1, 2])
            
            synd_op = Operator(qc)

            # Syndrome measurement
            synd_res, matrix = matrix.evolve(synd_op).measure([1, 2])
            matrix = partial_trace(matrix, [1, 2])
            
            cor_op = []
            for i in range(3):
                if not i:
                    theta = arctan((1-gamma)**2)
                    qc = QuantumCircuit(2)
                    qc.cx(1, 0)
                    qc.ry(2*theta, 1)
                    qc.cry(pi/2-2*theta, 0, 1)
                    cor_op.append(Operator(qc))
                elif not ~-i:
                    theta = arccos(1-gamma)
                    qc = QuantumCircuit(3)
                    qc.x(2)
                    qc.cry(theta, 2, 0)
                    cor_op.append(Operator(qc))
                elif not ~-~-i:
                    theta = arccos(1-gamma)
                    qc = QuantumCircuit(3)
                    qc.x(1)
                    qc.cry(theta, 1, 0)
                    cor_op.append(Operator(qc))
            
            # Apply correction operation
            if not int(synd_res, 2):
                matrix = matrix.evolve(cor_op[int(synd_res, 2)])
                matrix = partial_trace(matrix.measure([1])[1], [1])
                fid = state_fidelity(DensityMatrix(state), matrix)
            elif not ~-int(synd_res, 2):
                matrix = DensityMatrix([1, 0]).expand(matrix).evolve(cor_op[int(synd_res, 2)])
                matrix = partial_trace(matrix.measure([0])[1], [0, 1])
                fid = state_fidelity(DensityMatrix(state), matrix)
            elif not ~-~-int(synd_res, 2):
                matrix = DensityMatrix([1, 0]).expand(matrix).evolve(cor_op[int(synd_res, 2)])
                matrix = partial_trace(matrix.measure([0])[1], [0, 2])
                fid = state_fidelity(DensityMatrix(state), matrix)
            else:
                fid = 0

            fid_QEC4[param] += fid/NUM
            itr = ~-itr
        except:
            continue
    
    # SDP
    enc_op = Choi(enc_op)
    fid = 1

    # Create noise operators [Step 1]
    noise_ops = damp_err(gamma, num_qubit)
    nops = []
    for i in range(2**num_qubit):
        nops.append([])
        for j in range(2**num_qubit):
            nops[i].append([])
            nop = zeros((2**num_qubit, 2**num_qubit))
            for noise_op in noise_ops:
                nop += outer(noise_op.conjugate()[:,i], noise_op[:,j])
            nops[i][j] = nop
    noise_ops = nops

    # Create state operators [Step 2]
    state_ops = []
    for i in range(2**num_qubit):
        state_ops.append([])
        for j in range(2**num_qubit):
            state_ops[i].append([])
            state_ops[i][j] = outer(state_exp.data[:,i], state_exp.data.conjugate()[:,j])
            
    # Run SDP [Step 3-8]
    itr = 0
    while(True):
        itr += 1
        # SDP for Decoding
        # Define variable
        variable = Variable((2**(num_qubit<<1), 2**(num_qubit<<1)), hermitian=True)

        # Define constraints
        constraints = [variable >> 0, part_tr(variable, (2**num_qubit, 2**num_qubit), 1) == eye(2**num_qubit)]

        # Define objective function [Step 4]
        Obj = array(enc_op)[0][0]*kron(noise_ops[0][0], state_ops[0][0])
        for i in range(2**num_qubit):
            for j in range(2**num_qubit):
                for k in range(2**num_qubit):
                    for l in range(2**num_qubit):
                        Obj += array(enc_op)[l*2**num_qubit+j][k*2**num_qubit+i]*kron(noise_ops[i][j], state_ops[k][l])
        Obj -= array(enc_op)[0][0]*kron(noise_ops[0][0], state_ops[0][0])
        objective = real(trace(variable@Obj))
        
        # Solve SDP [Step 5]
        fidelity = Problem(Maximize(objective), constraints).solve()
        dec_op = variable.value
        if abs(fidelity - fid) < _ATOL:
            break
        fid = fidelity
            
        # SDP for Encoding
        # Define variable
        variable = Variable((2**(num_qubit<<1), 2**(num_qubit<<1)), hermitian=True)

        # Define constraints
        constraints = [variable >> 0, part_tr(variable, (2**num_qubit, 2**num_qubit), 1) == eye(2**num_qubit)]

        # Define objective function [Step 6]
        mat = zeros((2**(num_qubit<<1), 2**(num_qubit<<1)))
        mat[0][0] = 1
        Obj = tr(dec_op@kron(noise_ops[0][0], state_ops[0][0]))*mat
        for i in range(2**num_qubit):
            for j in range(2**num_qubit):
                for k in range(2**num_qubit):
                    for l in range(2**num_qubit):
                        mat = zeros((2**(num_qubit<<1), 2**(num_qubit<<1)))
                        mat[k*2**num_qubit+i][l*2**num_qubit+j] = 1
                        Obj += tr(dec_op@kron(noise_ops[i][j], state_ops[k][l]))*mat
        mat = zeros((2**(num_qubit<<1), 2**(num_qubit<<1)))
        mat[0][0] = 1
        Obj -= tr(dec_op@kron(noise_ops[0][0], state_ops[0][0]))*mat
        objective = real(trace(variable@Obj))
        
        # Solve SDP [Step 7]
        fidelity = Problem(Maximize(objective), constraints).solve()
        enc_op = variable.value
        if abs(fidelity - fid) < _ATOL:
            break
        fid = fidelity
    fid = fidelity
    fid_SDP4[param] += fid

    # Single qubit case
    E = damp_err(damp_params[param], 1)
    A = abs(tr(matmul(state, E[0])))**2
    for e in E[1:]:
        A += abs(tr(matmul(state.data, e)))**2
    fid_sing[param] += A

# Plot result
_, ax = subplots(1, 1)
#         ax.plot(damp_params, fid_QEC5, label = "[[5, 1, 3]]")
#         ax.plot(damp_params, fid_SDP5, label = "[[5, 1, 3]] SDP")
ax.plot(damp_params, fid_QEC4, label = "[[4, 1, 2]]")
ax.plot(damp_params, fid_SDP4, label = "[[4, 1, 2]] SDP", ls = '--')
ax.plot(damp_params, fid_sing, label = "Single qubit", ls = ':')
ax.set_xlabel(r'Damping probability $(\gamma)$')
ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
legend()
# show()
savefig(f'seq.png')
