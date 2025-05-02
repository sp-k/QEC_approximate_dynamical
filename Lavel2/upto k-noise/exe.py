from qiskit.quantum_info import DensityMatrix, random_statevector, random_density_matrix, Kraus
from numpy import trace, matmul, sqrt
from pickle import dump, load

from QEC_gen import n_qubit_code

num_qubit = int(input('Enter number of physical qubits in a logical qubit: '))
num_error = int(input('Enter number of erroneous qubits: '))
if num_error > num_qubit:
	raise ValueError(f'The number of erroneous qubits {num_error} cannot be more than the number of physical qubits {num_qubit}.')
num_mem = 1#int(input('Enter number of memory for adaptivity: '))
num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]
state = DensityMatrix([[0.5, 0], [0, 0.5]])

def damp_err(gamma, n=1):
    '''This method produces noise operators for a single-qubit amplitude damping channel with damping strength gamma.
    Arguments:
		gamma [float]: Damping probability
		n [int]: Number of qubit
	Returns:
		Kraus operators corresponding to the channel
    '''
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Number of qubit should be positive integer. Given {n}")
    if not isinstance(gamma, float) or gamma < 0 or gamma > 1:
        raise ValueError(f"Damping probability should lie between 0 and 1. Given {gamma}")
        
    from numpy import eye, kron, sqrt, zeros
    
    _E = [eye(2), zeros((2,2))]
    _E[0][1][1] = sqrt(1-gamma)
    _E[1][0][1] = sqrt(gamma)
    
    E = []
    for j in range(n):
        E.append(kron(eye(2**j), kron(_E[0]/sqrt(n), eye(2**(n-j-1)))))
        E.append(kron(eye(2**j), kron(_E[1]/sqrt(n), eye(2**(n-j-1)))))
    return E

def exec_QEC(state, i):
	'''This method executes the SDPs and calculates the fidelity.
	Arguments:
		state [<DensityMatrix>]: initial state (maximally mixed state)
		i [int]: Index of the damp_params to access damping strength value
	Returns: None
	'''
	from os import getpid
	print(f'{i}: {getpid()}')	# prints PID of the cirrent process

	# Calculates entanglement fidelity without encoding
	E = Kraus(damp_err(damp_params[i], 1))	# converts noise operators into Kraus object
	A = 0
	for a in range(2):
		for b in range(2):
			ij = [[0, 0], [0, 0]]
			ij[a][b] = 1
			A += DensityMatrix(ij).evolve(E).data[a][b].real
	'''abs(trace(matmul(state.data, E[0])))**2
	for e in E[1:]:
		A += abs(trace(matmul(state.data, e)))**2'''
	fid_sing[i] += A/4	# entanglement fidelity without encoding

	# SDP for n-qubit code
	QEC_n = n_qubit_code(num_qubit, num_mem, num_error)	# creates code object

	fid, enc, chk, dec = QEC_n.run_SDP(damp_params[i], state, 1e-2)	# runs the SDP
	fid_SDPn[i] += fid	# fidelity returns by the SDP
	print(i, damp_params[i], fid_SDPn[i], enc, chk, dec)	# prints damping parameter index, value, corresponding fidelity, encoding, check and decoding operatos in Choi form
	
	# Save the operations in a binary pickle file as a dictionary
	with open(f'choi_{num_qubit}{num_error}{i}', 'wb') as file:
		data = {'enc': enc, 'chk': chk, 'dec': dec}
		dump(data, file)

# Lists contain fidelities, initialized as 0.
fid_SDPn = [0 for _ in range(len(damp_params))]
fid_sing = [0 for _ in range(len(damp_params))]
for i in range(len(damp_params)):
	# Run the SDP for i-th damping parameters
	exec_QEC(state, i)
print(f'Fidelity from SDP where {num_error} out of {num_qubit} is (are) erroneous: {fid_SDPn}')



'''# To plot the fidelities against the damping strength
from matplotlib.pyplot import subplots, legend, savefig, show

_, ax = subplots(1, 1)
ax.plot(damp_params, fid_SDPn, label = f"{num_qubit}-qubit SDP", ls = '--')
ax.plot(damp_params, fid_sing, label = "Single qubit", ls = ':')
ax.set_xlabel(r'Damping probability $(\gamma)$')
ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
legend()
# savefig(f'{num_qubit}-qubit.png')
show()'''

