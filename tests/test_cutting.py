import numpy as np
from qiskit import QuantumCircuit

from quantum_forge import QuantumCircuitCutter


def make_circuit(num_qubits=6, layers=40, seed=0):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for _ in range(layers):
        for q in range(num_qubits):
            if rng.random() < 0.4:
                qc.ry(rng.uniform(0, np.pi), q)
        a, b = rng.choice(num_qubits, size=2, replace=False)
        qc.cx(int(a), int(b))
    return qc


def test_cuts_partition_gates_in_order():
    qc = make_circuit()
    cutter = QuantumCircuitCutter(max_subcircuit_width=4, max_subcircuit_depth=30)
    subs = cutter.cut_circuit(qc)

    assert len(subs) >= 2
    assert sum(len(s.circuit.data) for s in subs) == len(qc.data)

    # positions strictly increasing and every subcircuit non-empty
    positions = [c.position for c in cutter.cut_points]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    assert all(len(s.circuit.data) > 0 for s in subs)


def test_subcircuit_gates_match_original_after_remapping():
    qc = make_circuit()
    cutter = QuantumCircuitCutter(max_subcircuit_depth=30)
    subs = cutter.cut_circuit(qc)

    original = [
        (inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits), tuple(inst.operation.params))
        for inst in qc.data
    ]
    rebuilt = []
    for s in subs:
        for inst in s.circuit.data:
            local = tuple(s.circuit.find_bit(q).index for q in inst.qubits)
            rebuilt.append(
                (inst.operation.name, tuple(s.original_qubits[i] for i in local), tuple(inst.operation.params))
            )
    assert rebuilt == original


def test_subcircuits_only_use_touched_qubits():
    qc = make_circuit()
    cutter = QuantumCircuitCutter(max_subcircuit_depth=30)
    for s in cutter.cut_circuit(qc):
        touched = set()
        for inst in s.circuit.data:
            touched.update(s.circuit.find_bit(q).index for q in inst.qubits)
        assert touched == set(range(s.circuit.num_qubits))


def test_bitstring_mapping_is_little_endian():
    # local qubit 0 -> original 2, local qubit 1 -> original 0
    mapped = QuantumCircuitCutter._map_bitstring_to_original("01", {0: 2, 1: 0}, 3)
    # local "01": local q0 = '1', local q1 = '0'  ->  original q2 = '1', q0 = '0'
    assert mapped == "100"


def test_reconstruct_normalises():
    qc = make_circuit()
    cutter = QuantumCircuitCutter(max_subcircuit_depth=30)
    subs = cutter.cut_circuit(qc)
    fake = [{"0" * s.circuit.num_qubits: 10.0, "1" * s.circuit.num_qubits: 30.0} for s in subs]
    out = cutter.reconstruct_results(fake, qc.num_qubits)
    assert sum(out.values()) == 1.0 or abs(sum(out.values()) - 1.0) < 1e-9
    assert all(len(k) == qc.num_qubits for k in out)


def test_short_circuit_yields_single_subcircuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    subs = QuantumCircuitCutter().cut_circuit(qc)
    assert len(subs) == 1
    assert subs[0].input_qubits == set() and subs[0].output_qubits == set()
