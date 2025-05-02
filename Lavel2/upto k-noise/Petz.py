from numpy import zeros_like, zeros, outer, kron, conjugate, sqrt, dot, log2, eye, array
from scipy.linalg import sqrtm, pinv


num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]
# state = DensityMatrix([[0.5, 0], [0, 0.5]])

def damp_err_static(gamma, n, m):
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
    
	_E = [eye(2), zeros((2,2))]
	_E[0][1][1] = sqrt(1-gamma)
	_E[1][0][1] = sqrt(gamma)
    
	E = []
    
	for j in range(m+1):
	    id_ind = list(combinations(range(n), n-j))	# combinations of qubits without noise
	    for i in id_ind:
	        E_ = eye(1)
	        for k in range(n):
	            # Apply noise on appropriate qubits
	            if k in i:
	                # Apply no-error operation on qubit k
	                E_ = kron(E_, _E[0])
	            else:
	                # Apply error noise operation on qubit k
	                E_ = kron(E_, _E[1])
	        E.append(E_.copy())
	return E

class Petz:
	def __init__(self, code_basis, chk_basis = None):
		self.basis = code_basis
		self.chk_basis = chk_basis
		self._projector()
		pass
	
	def static_petz(self, noise_ops, all = False):
		EP = zeros_like(self.projector, dtype=complex)
		for noise in noise_ops:
		    EP += noise@self.projector@noise.conj().T
		EP_inv_sqrt = pinv(sqrtm(EP))
		recovery_ops = [self.projector@noise.conj().T@EP_inv_sqrt for noise in noise_ops]
		return recovery_ops
	
	def single_chk_petz(self, noise_ops, all = False):
		raise NotImplementedError()
	
	def _projector(self):
		P = zeros((len(self.basis[0]), len(self.basis[0])), dtype=complex)
		for vec in self.basis:
			P += outer(vec, conjugate(vec))
		self.projector = P
		pass

def ent_fid(base, recovery_ops, noise_ops):
	dim = len(base)
	fid = 0
	for rec in recovery_ops:
		for noise in noise_ops:
			tr = dot(base[0]@rec@noise, base[0])
			for basis in base[1:]:
				tr += dot(basis@rec@noise, basis)
			fid += (tr*tr.conj()).real
	return fid/(dim**2)

def code_51():
	'''[5, 1] code from arXiv:2410.00155'''
	basis = [zeros(2**5), zeros(2**5)]
	basis[0][24] = 1/sqrt(10)
	basis[0][20] = 1/sqrt(10)
	basis[0][18] = 1/sqrt(10)
	basis[0][17] = 1/sqrt(10)
	basis[0][12] = 1/sqrt(10)
	basis[0][10] = 1/sqrt(10)
	basis[0][9] = 1/sqrt(10)
	basis[0][6] = 1/sqrt(10)
	basis[0][5] = 1/sqrt(10)
	basis[0][3] = 1/sqrt(10)
	basis[1][-1] = 1
	return basis

def code_41():
	'''[4, 1] code from PhysRevA.56.2567'''
	basis = [zeros(2**4), zeros(2**4)]
	basis[0][0] = 1/sqrt(2)
	basis[0][-1] = 1/sqrt(2)
	basis[1][3] = 1/sqrt(2)
	basis[1][12] = 1/sqrt(2)
	return basis

def code_31():
	'''[3, 1] code from arXiv:2410.00155'''
	basis = [zeros(2**3), zeros(2**3)]
	basis[0][4] = 1/sqrt(3)
	basis[0][2] = 1/sqrt(3)
	basis[0][1] = 1/sqrt(3)
	basis[1][-1] = 1
	return basis

fids = []

for m in range(3, 6):
	base = [code_31(), code_41(), code_51()]
	
	for basis in base:
		fids.append([])
		for gamma in damp_params:
			noise_ops = damp_err_static(gamma, int(log2(len(basis[0]))), m)
			if not noise_ops:
				continue
			petz = Petz(basis)
			recovery_ops = petz.static_petz(noise_ops)
			fids[-1].append(ent_fid(basis, recovery_ops, noise_ops))
		print(f'Fidelity for {m} out of {int(log2(len(basis[0])))} qubits with [{int(log2(len(basis[0])))}, 1] code: {fids[-1]}')
