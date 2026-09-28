# Legacy modules

These files are the original, unfinished drafts of the circuit optimizer,
dynamic compiler and error mitigator. They are **not part of the installable
package**, are not tested, and do not run: each calls several helper methods
that were never written, and they target the pre-1.0 Qiskit API.

They are kept here for reference only. Mature implementations of everything
they sketch already exist:

- circuit optimisation and routing: the Qiskit transpiler, `pytket`
- error mitigation (ZNE, PEC, measurement calibration): `mitiq`
- pulse-level compilation: Qiskit Pulse / Qiskit Dynamics

The plan is to replace these with thin adapters that hand a `QuantumForgeEnv`
circuit to those libraries rather than reimplementing them.
