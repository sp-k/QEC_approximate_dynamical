# QEC_SDP

In `1-tensor`:
  1. `QEC.py` contains the static see-saw with Algorithm 4.
  2. `execute.py` executes the codes in QEC and produces the fidelities.
  3. `plot.py` plots the fidelities produced by `execute.py` as `FigStatic.pdf`.

In `Level2/m-qubit Noise`:
  1. `QEC_gen.py` contains the see-saw as in Algorithm 1.
  2. `exe.py` executes the see-saw algorithm and produces the fidelities along with the corresponding encoder, decoder and check instruments.
  3. `Petz.py` executes the [5, 1] and [3, 1] codes from Ref. arXiv:2410.00155 and the [4, 1] code from Ref. PhysicalReviewA.56(4):2567 and produces the fidelities.
  4. `plot.py` plots the fidelities produced by `exe.py` and `Petz.py`.
  5. `fid_res.txt` contains the fidelities produced by `exe.py`.
  6. `choi_XY` contains the Choi forms of the encoder, decoder and check instruments for X-qubit code with Y-qubit error given by `exe.py` in pickle format (as a dictionary).
  7. `Sanity_ch.py` checks the validity of the choi operations in `choi_XY` files.
