import Mathlib

/-!
The real-variable inequalities behind Theorem 1 of the paper.

For the chain on `{0, 1, 2}` that moves from `i` to `i - 1` at rate `1` and from `i` to `i + 1` at
rate `ε`, Section 2 of the paper shows that for `0 ≤ t ≤ 2` the largest variation distance `d̄(t)`
satisfies `d̄(t) ≥ e^{-εt} - 1 + g((1 + ε) t)`, where `g(s) = e^{-s} (1 + s)`.  The theorem
`lower_bound_gt` shows that the right-hand side exceeds `e^{-1}` when `0 < ε ≤ 1/100`.
-/

/-- `g(s) = e^{-s} (1 + s)` is nonincreasing on `[0, ∞)`. -/
theorem g_antitone {s S : ℝ} (hs : 0 ≤ s) (hsS : s ≤ S) :
    Real.exp (-S) * (1 + S) ≤ Real.exp (-s) * (1 + s) := by
  have hsplit : Real.exp (-s) = Real.exp (-S) * Real.exp (S - s) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hexp : 1 + (S - s) ≤ Real.exp (S - s) := by
    have := Real.add_one_le_exp (S - s)
    linarith
  have hpos : 0 < Real.exp (-S) := Real.exp_pos _
  have h1 : (1 + (S - s)) * (1 + s) ≤ Real.exp (S - s) * (1 + s) :=
    mul_le_mul_of_nonneg_right hexp (by linarith)
  have h2 : 1 + S ≤ (1 + (S - s)) * (1 + s) := by
    nlinarith [mul_nonneg hs (sub_nonneg.mpr hsS)]
  rw [hsplit]
  calc Real.exp (-S) * (1 + S) ≤ Real.exp (-S) * (Real.exp (S - s) * (1 + s)) :=
        mul_le_mul_of_nonneg_left (h2.trans h1) hpos.le
    _ = Real.exp (-S) * Real.exp (S - s) * (1 + s) := by ring

/-- The margin `3 - e - ε (4 + 2 e²) - 4 ε² > 0` for `0 < ε ≤ 1/100`. -/
theorem margin {ε : ℝ} (h0 : 0 < ε) (h1 : ε ≤ 1 / 100) :
    0 < 3 - Real.exp 1 - ε * (4 + 2 * Real.exp 1 ^ 2) - 4 * ε ^ 2 := by
  have he : Real.exp 1 < 2718281829 / 1000000000 :=
    Real.exp_one_lt_d9.trans (by norm_num)
  have he0 : 0 < Real.exp 1 := Real.exp_pos 1
  have he2 : Real.exp 1 ^ 2 < 739 / 100 := by
    nlinarith [mul_pos he0 (sub_pos.mpr he)]
  nlinarith [mul_nonneg h0.le (sub_nonneg.mpr he2.le), mul_nonneg h0.le (sub_nonneg.mpr h1)]

/-- For `0 < ε ≤ 1/100` and `0 ≤ t ≤ 2`, `e^{-εt} - 1 + g((1 + ε) t) > e^{-1}`. -/
theorem lower_bound_gt {ε t : ℝ} (h0 : 0 < ε) (h1 : ε ≤ 1 / 100) (ht0 : 0 ≤ t) (ht2 : t ≤ 2) :
    Real.exp (-1) <
      Real.exp (-(ε * t)) - 1 + Real.exp (-((1 + ε) * t)) * (1 + (1 + ε) * t) := by
  have hx : 1 - 2 * ε ≤ Real.exp (-(ε * t)) := by
    have := Real.add_one_le_exp (-(ε * t))
    nlinarith [mul_le_mul_of_nonneg_left ht2 h0.le]
  have hg : Real.exp (-(2 + 2 * ε)) * (1 + (2 + 2 * ε))
      ≤ Real.exp (-((1 + ε) * t)) * (1 + (1 + ε) * t) :=
    g_antitone (mul_nonneg (by linarith) ht0)
      (by nlinarith [mul_le_mul_of_nonneg_left ht2 (show (0 : ℝ) ≤ 1 + ε by linarith)])
  have hsplit : Real.exp (-(2 + 2 * ε)) = Real.exp (-2) * Real.exp (-(2 * ε)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hy : 1 - 2 * ε ≤ Real.exp (-(2 * ε)) := by
    have := Real.add_one_le_exp (-(2 * ε))
    linarith
  have hz : 0 < Real.exp (-2) := Real.exp_pos _
  have hy' : Real.exp (-2) * (1 - 2 * ε) * (3 + 2 * ε)
      ≤ Real.exp (-2) * Real.exp (-(2 * ε)) * (3 + 2 * ε) :=
    mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hy hz.le) (by linarith)
  have hm1 : Real.exp (-1) = Real.exp 1 * Real.exp (-2) := by
    rw [← Real.exp_add]
    norm_num
  have hz2 : Real.exp (-2) * Real.exp 1 ^ 2 = 1 := by
    rw [sq, ← Real.exp_add, ← Real.exp_add]
    norm_num
  have hzε : Real.exp (-2) * Real.exp 1 ^ 2 * ε = ε := by
    rw [hz2, one_mul]
  have hmz : 0 < Real.exp (-2) * (3 - Real.exp 1 - ε * (4 + 2 * Real.exp 1 ^ 2) - 4 * ε ^ 2) :=
    mul_pos hz (margin h0 h1)
  rw [hsplit] at hg
  rw [hm1]
  nlinarith [hg, hy', hx, hmz, hzε]
