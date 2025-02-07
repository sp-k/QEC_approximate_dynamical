from qiskit.quantum_info import partial_trace, DensityMatrix
from numpy.linalg import eigvals
from numpy import array, all
from pickle import load
with open('choi_41', 'rb') as file:
    data = load(file)

num_qubit = 4

for op in data:
    if op == 'enc':
        mat = array(data[op])
    else:
        mat = data[op][0]
    print(partial_trace(DensityMatrix(mat), range(num_qubit)))
    print(all(eigvals(mat)>-1e-4))
