# The variation threshold, Kemeny's constant and stationary times of reversible Markov chains

Companion material for the paper of the same title (Y. Shmalo, October 2026). [Read the paper](paper/main.pdf).

For a finite irreducible reversible Markov chain in continuous time with stationary distribution π, Chapter 4 of D.
Aldous and J. A. Fill, *Reversible Markov Chains and Random Walks on Graphs*, compares several parameters, among them

- the average hitting time (Kemeny's constant) τ₀ = Σ_{i,j} π_i π_j E_i T_j,
- the variation threshold τ₁ = min{t : d̄(t) ≤ e⁻¹}, where d̄(t) is the largest total variation distance between the
  laws at time t of the chain started at two different states,
- the relaxation time τ₂ = 1/λ₂, where λ₂ is the spectral gap,
- the stationary-time parameter τ₁⁽²⁾ = max_i min_U E_i U, the minimum over (randomized) stopping times U such that
  X_U has distribution π under P_i.

Among the inequalities proved there are τ₂ ≤ τ₁ ≤ 66 τ₀ and τ₂ ≤ τ₀. Open Problem 4.48 of the book asks:
(a) is τ₁ ≤ τ₀? (b) is τ₂ ≤ τ₁⁽²⁾? The paper answers (a) negatively and (b) affirmatively.

- Theorem 1. For 0 < ε ≤ 1/100, the chain on {0, 1, 2} that jumps from i to i − 1 at rate 1 and from i to i + 1 at
  rate ε has τ₀ < 2 < τ₁. For the same chain on {0, 1, ..., k}, τ₁/τ₀ → t_k/k as ε → 0, where P(E₁ + ... + E_k >
  t_k) = e⁻¹ for independent standard exponential variables E_j; t₄/4 = 1.08799... .
- Theorem 2. Every finite irreducible reversible chain satisfies τ₂ ≤ τ₁⁽²⁾. If u is an eigenfunction for the spectral
  gap and |u| is largest at the state i, then every stopping time U such that X_U has distribution π under P_i
  satisfies E_i U ≥ τ₂.

Both statements hold for discrete-time chains with the conventions of the book. The best constant C in τ₁ ≤ C τ₀ lies
between t₄/4 and 66.

## Contents

- `paper/main.tex`, `paper/main.pdf`: the paper.
- `code/checks.py`: numerical checks of the identities and inequalities used in the proofs, in multiple-precision
  arithmetic where it matters (Python 3 with mpmath, NumPy and SciPy). `code/checks_output.txt` is its output.
- `code/Margin.lean`: the real-variable inequalities in the proof of Theorem 1(i), proved in Lean 4 with Mathlib.
  `code/lean_output.txt` records the compiler's exit status.
- `SHA256SUMS.txt`: checksums of the files above.

## Running the checks

    python code/checks.py > code/checks_output.txt

The Lean file was checked with Lean 4.33.1 and Mathlib commit
0df444a360eaa60ab8c11dca51a86af692955474; inside a Lake project that depends on that Mathlib, run

    lake env lean Margin.lean
