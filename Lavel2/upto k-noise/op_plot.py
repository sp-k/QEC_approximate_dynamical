'''from qiskit.quantum_info import DensityMatrix, state_fidelity
from matplotlib.pyplot import subplots, show
from math import sqrt

def final_state(gamma):
    tr = sqrt(1-gamma) + 0.5*sqrt(gamma)*(sqrt(1-gamma)+sqrt(gamma)+1)
    return [[0.5, sqrt(gamma)*(sqrt(1-gamma)-sqrt(gamma)-1)*0.25/tr], [sqrt(gamma)*(sqrt(1-gamma)-sqrt(gamma)-1)*0.25/tr, 0.5]]

num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]
init_rho = DensityMatrix([[0.5, 0], [0, 0.5]])
fid = [state_fidelity(init_rho, DensityMatrix(final_state(gamma))) for gamma in damp_params]

_, ax = subplots(1, 1)
ax.plot(damp_params, fid)
show()'''

from numpy import array, kron, eye, real, sqrt, where, array2string, inf
from numpy.linalg import eig
from itertools import product
from qiskit.quantum_info import partial_trace, DensityMatrix, Operator

def L_op(ch_mat):
    num_qubit = 4
    state_init = [DensityMatrix([[1, 0], [0, 0]]), DensityMatrix([[0, 0], [0, 1]])]
    for sts in product(state_init, repeat = num_qubit):
        state = sts[0]
        for st in list(sts)[1:]:
            state = kron(state, st)
        I = eye(2**num_qubit)
        dm = partial_trace(ch_mat.data@kron(state, I), range(num_qubit, num_qubit<<1))
        print(dm, dm.purity(), dm.is_valid())

def ChoiToOp(ch_mat):
	num_qubit = 4
	e_val, e_vec = eig(ch_mat)
	il = where(e_val.real > 1e-2)[0]
# 	print(e_val.real, il)
	vecl = []
	for i in il:
		vec = sqrt(e_val[i].real)*e_vec[0:,i:i+1].real
		vecl.append(vec)
# 		print(vec)
		op = Operator(vec.reshape((2**num_qubit, 2**num_qubit)))
# 		print(op, op.is_unitary())
# 	for v1 in vecl:
# 		for v2 in vecl:
# 			print(v1.T@v2)
	vecl = []
	i = where(e_val == max(e_val))[0][0]
# 	print(e_val.real, i)
# 	i = 0
	vec = sqrt(e_val[i].real)*e_vec[0:,i:i+1].real
# 	print(vec)
	op = Operator(vec.reshape((2**num_qubit, 2**num_qubit)).T.real)
	return op, op.is_unitary()
'''    i = 0
    v = real(ch_mat.data)[i][i]
    while v < 0.1:
        i += 1
        v = real(ch_mat.data)[i][i]
    vec = [val/sqrt(v) for val in array(ch_mat.data)[i]]
    num_qubit = 3
    return Operator(array([vec[i:i+2**num_qubit] for i in range(0, 2**(num_qubit<<1), 2**num_qubit)]).T)'''

from pickle import load
with open('choi_414', 'rb') as file:
    data = load(file)
for op in data:
    if op == 'enc':
        print(op, ChoiToOp(data[op]))
        print(array2string(data[op], threshold = inf))
        L_op(data[op])
    else:
        print(op, ChoiToOp(data[op][0]))
        print(array2string(data[op][0], threshold = inf))
        L_op(data[op][0])
    input()

