from qiskit.quantum_info import DensityMatrix, random_statevector, random_density_matrix
from numpy import trace, matmul, sqrt
from pickle import dump, load

from QEC_gen_new import n_qubit_code

DEBUG = True
num_qubit = 4
num_mem = 1
num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]
state = DensityMatrix([[0.5, 0], [0, 0.5]])

if not DEBUG:
	if input("Want to rewrite file? [y/n]") != 'y':
		exit()
	try:
		with open('choi', 'rb') as file:
			data = load(file)
	except:
		with open('choi', 'wb') as file:
			data = {}
			dump(data, file)

	with open('choi', 'wb') as file:
		if f'{num_qubit}-qubit' not in list(data.keys()):
			data[f'{num_qubit}-qubit'] = {}
		if '2-tensor' not in list(data[f'{num_qubit}-qubit'].keys()):
			data[f'{num_qubit}-qubit']['2-tensor'] = {}
		if num_mem not in list(data[f'{num_qubit}-qubit']['2-tensor'].keys()):
			data[f'{num_qubit}-qubit']['2-tensor'][num_mem] = {}
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['enc'] = []
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['chk'] = []
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['dec'] = []
		dump(data, file)

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

def exec_QEC(state, i):
	from os import getpid
	print(f'{i}: {getpid()}')
	QEC_n = n_qubit_code(num_qubit, num_mem)

	fid, enc, chk, dec = QEC_n.run_SDP(damp_params[i], state, 1e-3)
	if not DEBUG:
		with open('choi', 'rb') as file:
			data = load(file)
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['enc'].append(enc)
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['chk'].append(chk)
		data[f'{num_qubit}-qubit']['2-tensor'][num_mem]['dec'].append(dec)
		with open('choi', 'wb') as file:
			dump(data, file)
	fid_SDPn[i] += fid
	print(i, damp_params[i], fid_SDPn[i], enc, chk, dec)

	E = damp_err(damp_params[i], 1)
	A = abs(trace(matmul(state.data, E[0])))**2
	for e in E[1:]:
		A += abs(trace(matmul(state.data, e)))**2
	fid_sing[i] += A
	if i == 9:
		with open('choi_temp', 'wb') as file:
			data = {'enc': enc, 'chk': chk, 'dec': dec}
			dump(data, file)

fid_SDPn = [0 for _ in range(len(damp_params))]
fid_sing = [0 for _ in range(len(damp_params))]
for i in range(len(damp_params)):
    exec_QEC(state, i)
print(fid_SDPn)


from matplotlib.pyplot import subplots, legend, savefig, show

_, ax = subplots(1, 1)
ax.plot(damp_params, fid_SDPn, label = f"{num_qubit}-qubit SDP", ls = '--')
ax.plot(damp_params, fid_sing, label = "Single qubit", ls = ':')
ax.set_xlabel(r'Damping probability $(\gamma)$')
ax.set_ylabel(r'$F_e(\rho,(\mathcal{R}\circ\mathcal{E}))$')
legend()
# savefig(f'{num_qubit}-qubit.png')
show()
