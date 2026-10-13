# Processor Architecture Analysis

## Processor A (5 physical qubits)

Coupling graph is a sparse tree-like topology:

```
0 – 1 – 2
    |
    3 – 4
```

With only 5 qubits and limited connectivity, a 6-qubit logical circuit necessarily requires qubit reuse or aggressive routing, which typically increases SWAP count and depth.

## Processor B (9 physical qubits)

Richer connectivity that includes a central hub around qubit 2/6 and a ring-like structure:

```
0 – 1 – 2 – 3 – 4
|       | \ | /
8 – 6 – 5   |
    |
    7
```

(edges as defined in `processor_B.json`). The extra physical qubits and additional edges give the transpiler more freedom to find lower-depth embeddings and fewer SWAPs for a 6-qubit logical circuit.

## Expected Trade-offs

- Processor B has a larger physical budget → lower routing overhead expected.
- Processor A is more constrained → higher SWAP insertion and potentially larger depth.
- Actual measured depth, CX and SWAP counts are reported in `results/tables/circuit_resources.csv` after execution; do not assume B is always superior without looking at the numbers.

## Noise

Identical noise model (p1 = 0.001, pCX = 0.005, readout = 0.01) is applied for fair comparison. Deeper circuits on the more constrained architecture accumulate more noise, which can degrade classification metrics under noisy simulation.
