from matplotlib.pyplot import subplots, legend, show, rcParams, savefig

params = {
          'figure.figsize': (12, 6),
          'legend.fontsize': 15,
          'axes.labelsize': 15,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
         }
rcParams.update(params)

num_params = 10
damp_params = [i/(2*~-num_params) for i in range(num_params)]

# No Encoding fidelity
fis = [1.0, 0.9720237690148861, 0.9436267430132539, 0.9147687979209719, 0.8854029962885428, 0.8554738483549543, 0.8249149571305296, 0.7936457577630638, 0.7615668851388537, 0.7285533905932737]

# Single-qubit error
QEC5s = [1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000007, 1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000002, 1.0000000000000007, 1.0000000000000002]
SDP5s = [1.0000002549183629, 0.9997671640728529, 1.0000002300359045, 0.9999840565349504, 0.9999960672176433, 1.0001129441080925, 0.9999999750827795, 0.9999744824008846, 1.0000048230544263, 0.9999988395145082]
QEC4s = [1.0000000000000007, 0.9999965832058367, 0.9968992359762094, 0.9868011380873779, 0.9825187039681598, 0.9740350724829361, 0.9736563112068658, 0.9630985247466158, 0.9403941522125782, 0.9156290337059805]
SDP4s = [1.0000027111704957, 0.9999503676041123, 0.9997955265638233, 0.9995351523267824, 0.9991401074173436, 0.998586588633571, 0.9978772727797605, 0.9970164530489534, 0.9959260304000603, 0.994600306954321]

# All-qubit error
QEC5a = [1-1.166*gamma**2 for gamma in damp_params]          # optimal recovery mentioned in PhysRevA.75.012338
SDP5a = [0.9999994298140122, 0.9970286937957573, 0.9883103821922707, 0.9737696276266068, 0.9536410895719232, 0.9282743457811966, 0.8981346397752208, 0.8638069536899622, 0.8260386259557286, 0.7855512674753571]
QEC4a = [1-1.25*gamma**2 for gamma in damp_params]          # optimal recovery mentioned in PhysRevA.75.012338
SDP4a = [1.0000027111705263, 0.9961432518104206, 0.9846151030111905, 0.9655106171613803, 0.9391284566949196, 0.9059231469180817, 0.8664304089032289, 0.8216944928685231, 0.7729667813552271, 0.7258206000691018]

_, ax = subplots(1, 1)
ax.plot(damp_params, fis, label = f"No Encoding")
ax.plot(damp_params, QEC5s, label = f"[[5, 1, 3]] QEC/ See-saw, 1-error")
ax.plot(damp_params, QEC4s, label = f"[[4, 1]] Approximate QEC, 1-error", ls = '--')
ax.plot(damp_params, SDP4s, label = f"[[4, 1]] See-saw, 1-error", ls = '--')

ax.plot(damp_params, QEC5a, label = f"[[5, 1, 3]] QEC, all error")
ax.plot(damp_params, SDP5a, label = f"[[5, 1, 3]] See-saw, all error")
ax.plot(damp_params, QEC4a, label = f"[[4, 1]] Approximate QEC, all error", ls = '--')
ax.plot(damp_params, SDP4a, label = f"[[4, 1]] See-saw, all error", ls = '--')

ax.set_xlabel(r'Damping probability $\gamma$')
ax.set_ylabel(r'Entanglement fidelity $F_{ent}$')
legend()
savefig("FigStatic.pdf", format="pdf", bbox_inches="tight")
show()
