import importlib.util

import gymnasium as gym
import numpy as np
import pytest

import quantum_forge  # noqa: F401  (registers QuantumForge-v0)
from quantum_forge import HardwareConstraints, QuantumForgeEnv, OPERATIONS

HAS_CIRQ = importlib.util.find_spec("cirq") is not None
HAS_AER = importlib.util.find_spec("qiskit_aer") is not None

BACKENDS = ["qiskit"] + (["cirq"] if HAS_CIRQ else [])


def ghz_actions():
    # H on 0, CX 0->1, CX 1->2
    return [
        [OPERATIONS.index("h"), 0, 0, 0.0, 0.0],
        [OPERATIONS.index("cx"), 0, 1, 0.0, 0.0],
        [OPERATIONS.index("cx"), 1, 2, 0.0, 0.0],
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_reset_returns_zero_state(backend):
    env = QuantumForgeEnv(num_qubits=2, backend=backend)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (8,)
    np.testing.assert_allclose(obs[:4], [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(obs[4:], 0, atol=1e-6)
    assert info["gate_count"] == 0
    assert info["circuit_depth"] == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_ghz_reaches_target(backend):
    env = QuantumForgeEnv(num_qubits=3, backend=backend, target="ghz")
    env.reset(seed=0)
    for action in ghz_actions():
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    assert info["fidelity"] == pytest.approx(1.0, abs=1e-6)
    assert terminated is True
    assert truncated is False
    assert info["gate_count"] == 3
    assert info["two_qubit_gate_count"] == 2
    assert info["circuit_depth"] == 3
    # reward is fidelity minus a small cost; must be positive for a perfect prep
    assert 0.5 < reward <= 1.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_agree_on_parameterised_gates(backend):
    """Every parameterised operation must produce the same state in both backends."""
    if backend != "qiskit":
        pytest.skip("comparison runs from the qiskit parametrisation only")
    if not HAS_CIRQ:
        pytest.skip("cirq not installed")
    rng = np.random.default_rng(1)
    for op in range(len(OPERATIONS)):
        actions = [[op, 0, 1, rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)] for _ in range(2)]
        # Add some non-trivial prior state
        actions = [[OPERATIONS.index("h"), 0, 0, 0, 0], [OPERATIONS.index("ry"), 1, 0, 0.7, 0]] + actions
        states = []
        for b in ("qiskit", "cirq"):
            env = QuantumForgeEnv(num_qubits=2, backend=b, target="ghz")
            env.reset(seed=0)
            for a in actions:
                env.step(np.array(a, dtype=np.float32))
            states.append(env._backend.statevector())
        overlap = abs(np.vdot(states[0], states[1])) ** 2
        assert overlap == pytest.approx(1.0, abs=1e-5), f"backends disagree on op {OPERATIONS[op]}"


def test_random_target_is_seeded_and_normalised():
    env = QuantumForgeEnv(num_qubits=3)
    env.reset(seed=123)
    t1 = env.target_state.copy()
    env.reset(seed=123)
    t2 = env.target_state.copy()
    env.reset(seed=124)
    t3 = env.target_state.copy()
    np.testing.assert_allclose(t1, t2)
    assert not np.allclose(t1, t3)
    assert np.linalg.norm(t1) == pytest.approx(1.0)


def test_truncation_at_max_steps():
    env = QuantumForgeEnv(num_qubits=2, max_steps=4, success_threshold=1.01)
    env.reset(seed=0)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert steps == 4
    assert truncated


def test_action_decoding_clips_and_fixes_duplicate_qubits():
    env = QuantumForgeEnv(num_qubits=3)
    env.reset(seed=0)
    op, q1, q2, p1, p2 = env._decode_action([99, -5, 7, 10, -10])
    assert op == len(OPERATIONS) - 1
    assert q1 == 0 and q2 == 2
    assert p1 == pytest.approx(np.pi) and p2 == pytest.approx(-np.pi)
    op, q1, q2, _, _ = env._decode_action([OPERATIONS.index("cx"), 1, 1, 0, 0])
    assert q1 != q2


def test_hardware_constraints_reject_disconnected_pairs():
    hc = HardwareConstraints(
        connectivity_map={0: [1], 1: [0, 2], 2: [1]},
        gate_times={"cx": 300.0, "h": 50.0},
        max_parallel_gates=2,
    )
    env = QuantumForgeEnv(num_qubits=3, hardware_constraints=hc, target="ghz", max_violations=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (2 * 8 + 3,)

    # CX between 0 and 2 is not allowed
    obs, reward, terminated, truncated, info = env.step(np.array([OPERATIONS.index("cx"), 0, 2, 0, 0]))
    assert reward == -1.0
    assert info["hardware_violations"] == 1
    assert info["gate_count"] == 0
    assert not terminated

    # second violation exceeds max_violations=1
    _, _, terminated, _, _ = env.step(np.array([OPERATIONS.index("cx"), 2, 0, 0, 0]))
    assert terminated

    # allowed sequence accumulates execution time and busy flags
    env.reset(seed=0)
    env.step(np.array([OPERATIONS.index("h"), 0, 0, 0, 0]))
    obs, _, _, _, info = env.step(np.array([OPERATIONS.index("cx"), 0, 1, 0, 0]))
    assert info["estimated_execution_time"] == pytest.approx(350.0)
    busy = obs[-3:]
    np.testing.assert_array_equal(busy, [1, 1, 0])


def test_max_parallel_gates_increases_depth():
    hc = HardwareConstraints(connectivity_map={i: [] for i in range(4)}, max_parallel_gates=2)
    env = QuantumForgeEnv(num_qubits=4, hardware_constraints=hc)
    env.reset(seed=0)
    for q in range(4):
        _, _, _, _, info = env.step(np.array([OPERATIONS.index("h"), q, 0, 0, 0]))
    assert info["circuit_depth"] == 2  # 4 single-qubit gates, 2 per layer

    env_free = QuantumForgeEnv(num_qubits=4)
    env_free.reset(seed=0)
    for q in range(4):
        _, _, _, _, info = env_free.step(np.array([OPERATIONS.index("h"), q, 0, 0, 0]))
    assert info["circuit_depth"] == 1


def test_explicit_target_vector_validation():
    with pytest.raises(ValueError):
        QuantumForgeEnv(num_qubits=2, target=np.array([1, 1, 0, 0]))
    with pytest.raises(ValueError):
        QuantumForgeEnv(num_qubits=2, target=np.array([1, 0]))
    target = np.array([0, 1, 0, 0], dtype=complex)
    env = QuantumForgeEnv(num_qubits=2, target=target)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.array([OPERATIONS.index("x"), 0, 0, 0, 0]))
    assert info["fidelity"] == pytest.approx(1.0)
    assert terminated


def test_registered_with_gymnasium_and_passes_env_checker():
    from gymnasium.utils.env_checker import check_env

    env = gym.make("QuantumForge-v0", num_qubits=2)
    check_env(env.unwrapped, skip_render_check=True)


@pytest.mark.skipif(not HAS_AER, reason="qiskit-aer not installed")
def test_noisy_qiskit_fidelity_is_below_ideal():
    env = QuantumForgeEnv(num_qubits=2, noise_level=0.05, target="ghz")
    env.reset(seed=0)
    env.step(np.array([OPERATIONS.index("h"), 0, 0, 0, 0]))
    _, _, _, _, info = env.step(np.array([OPERATIONS.index("cx"), 0, 1, 0, 0]))
    assert 0.5 < info["fidelity"] < 0.999


@pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")
def test_noisy_cirq_fidelity_is_below_ideal():
    env = QuantumForgeEnv(num_qubits=2, backend="cirq", noise_level=0.05, target="ghz")
    env.reset(seed=0)
    env.step(np.array([OPERATIONS.index("h"), 0, 0, 0, 0]))
    _, _, _, _, info = env.step(np.array([OPERATIONS.index("cx"), 0, 1, 0, 0]))
    assert 0.5 < info["fidelity"] < 0.999


def test_rgb_render_returns_image():
    pytest.importorskip("matplotlib")
    env = QuantumForgeEnv(num_qubits=2, render_mode="rgb_array")
    env.reset(seed=0)
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
