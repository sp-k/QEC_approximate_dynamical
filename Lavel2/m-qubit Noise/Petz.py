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
	basis[0][1] = 1/sqrt(2)
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

def our_code_2():
	basis = [zeros(2**2), zeros(2**2)]
	basis[0][0] = sqrt(0.5)
	basis[0][-1] = sqrt(0.5)
	basis[1][0] = sqrt(0.5)
	basis[1][-1] = -sqrt(0.5)
	chk = array([[sqrt(0.5), 0, sqrt(0.5), sqrt(0.5)], [0, 1, 0, 0], [sqrt(0.5), 0, -sqrt(0.5), -sqrt(0.5)], [0, 0, 0, 0]]).data
	return basis, [chk@basis[0], chk@basis[1]]

def our_code_3():
	basis = [zeros(2**3), zeros(2**3)]
	basis[0][0] = sqrt(0.5)
	basis[0][-1] = sqrt(0.5)
	basis[1][0] = sqrt(0.5)
	basis[1][-1] = -sqrt(0.5)
	chk = array([[1, 0, 0, 0, 0, 0, 0, 0],
										   [0, 1, 0, 0, 0, 0, 0, 0],
										   [0, 0, 1, 0, 0, 0, 0, 0],
										   [0, 0, 0, 1, 0, 0, 0, 0],
										   [0, 0, 0, 0, 1, 0.5, 0.5, 1], 
										   [0, 0, 0, 0, 0, 0.5, 0, 0],
										   [0, 0, 0, 0, 0, 0, 0.5, 0],
										   [0, 0, 0, 0, 0, 0, 0, 0]]).data
	return basis, [chk@basis[0], chk@basis[1]]

def our_code_4():
	basis = [zeros(2**4), zeros(2**4)]
	basis[0][0] = 1/sqrt(2)
	basis[0][-1] = 1/sqrt(2)
	basis[1][0] = 1/sqrt(2)
	basis[1][-1] = -1/sqrt(2)
	chk = array([[1, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0, 0],
										   [0, 1, 0, 1/3, 0, 1/3,    0,    0, 0, 1/3,    0,    0,    0,    0,    0, 0],
										   [0, 0, 1, 1/3, 0,    0, 1/3,    0, 0,    0, 1/3,    0,    0,    0,    0, 0],
										   [0, 0, 0, 1/3, 0,    0,    0, 0.5, 0,    0,   0, 0.5,    0,    0,    0, 0],
										   [0, 0, 0,    0, 1, 1/3, 1/3,    0, 0,    0,    0,    0, 1/3,    0,    0, 0], 
										   [0, 0, 0,    0, 0, 1/3,    0, 0.5, 0,    0,    0,    0,    0, 0.5,    0, 0],
										   [0, 0, 0,    0, 0,    0, 1/3, 0.5, 0,    0,    0,    0,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0, 0.5, 0,    0,    0,    0,    0,    0,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 1, 1/3, 1/3,     0, 1/3,    0,    0, 1],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0, 1/3,    0, 0.5,    0, 1/3,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0, 1/3, 0.5,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0, 0.5,    0,    0,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0, 1/3, 1/3, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0, 1/3,    0, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0, 1/3, 0],
										   [0, 0, 0,    0, 0,    0,    0,    0, 0,    0,    0,    0,    0,    0,    0, 0]]).data
	return (basis, [chk@basis[0], chk@basis[1]])

fids = []

for m in range(1, 6):
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

	base = [our_code_2(), our_code_3(), our_code_4()]

	for basis in base:
		fids.append([])
		for gamma in damp_params:
			noise_ops = damp_err_static(gamma, int(log2(len(basis[0][0]))), m)
			if not noise_ops:
				continue
			petz = Petz(basis[0], basis[1])
			recovery_ops = petz.static_petz(noise_ops)
			fids[-1].append(ent_fid(basis[0], recovery_ops, noise_ops))
		print(f'Fidelity for {m} out of {int(log2(len(basis[0][0])))} qubits with our {int(log2(len(basis[0][0])))}-qubit code: {fids[-1]}')
# fids = [[0.9999999962002966, 0.9944911897880423, 0.9887296154240831, 0.9826916559564663, 0.9763489734709674, 0.9696677997883347, 0.9626097316970446, 0.955124994879582, 0.9471543734292172, 0.9386231417169694],
# [1.0000000000000002, 0.9963204086258307, 0.9925966212230022, 0.9886912050725045, 0.9845411541934934, 0.9800960642576845, 0.9753070792871036, 0.9701217931423038, 0.964480336019532, 0.9583111096647982]
# [1.0000000000000007, 0.990608438602521, 0.9809363471940227, 0.9709569763917598, 0.9606390345627308, 0.9499455285329335, 0.9388321936425761, 0.9272453199901906, 0.9151186641666437, 0.9023689270621824], 
# [0.9999999999999996, 0.9466368619897739, 0.8972135954999575, 0.8511485559356866, 0.8079529266996415, 0.7672090536041427, 0.7285533905932735, 0.6916624790355397, 0.6562406826268672, 0.622008467928146], 
# [0.9999999982186819, 0.9721977568291672, 0.9455280010016817, 0.9198113546203376, 0.8949907562154931, 0.8710329021973129, 0.8479124455339346, 0.8256058964898134, 0.8040879324872392, 0.7833280813760533], 
# [0.9999999957678296, 0.9876561923308275, 0.9764037161281789, 0.9657464184781704, 0.9555018692616184, 0.9455586886640286, 0.9358346653529999, 0.9262607854126398, 0.9167726427500279, 0.9073041854360504]]
