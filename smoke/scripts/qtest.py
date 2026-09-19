"""Which criterion actually identifies the native fold?

The matched 18-reference null shows that "within 1.0 A of one of 18 references"
is a WEAK criterion for a 10-mer: an arbitrary 18-reference compact set is
matched 25% of the time.  RMSD counts therefore cannot, on their own, establish
that the native fold was found.  Q -- the fraction of 1UAO's own native contacts
present -- is reference-specific and cannot be matched by a coincidentally
compact structure, so it is the discriminating statistic.
"""
import json
import numpy as np

OUT = "/home/sulcjo/.claude/jobs/b27cb9c3/tmp/results"
z = np.load(f"{OUT}/frame_metrics.npz", allow_pickle=True)
r, q, cv, yw = z["rmsd_bb_min18"], z["q"], z["cv1"], z["yw"]
N = r.size
compact = (cv >= 0.82) & (cv <= 0.93)

print("NATIVE CONTACT FRACTION Q (vs 1UAO's own 65 native contacts)")
print(f"{'population':42s} {'n':>9s} {'median Q':>9s} {'p99':>7s} {'max':>7s}")
for lab, m in [("all frames", np.ones(N, bool)),
               ("compact frames, CV1 in [0.82,0.93]", compact),
               ("frames with bb RMSD <= 1.5 A", r <= 1.5),
               ("frames with bb RMSD <= 1.0 A", r <= 1.0)]:
    print(f"{lab:42s} {int(m.sum()):9,d} {np.median(q[m]):9.3f} "
          f"{np.percentile(q[m],99):7.3f} {q[m].max():7.3f}")

print("\nQ is what separates the native fold from merely-compact structures:")
for thr in (0.5, 0.6, 0.7, 0.8, 0.85):
    n_all = int((q >= thr).sum())
    n_comp = int((q[compact] >= thr).sum())
    print(f"  Q >= {thr:.2f}: {n_all:7,d} frames overall ({n_all/N:.2e}); "
          f"{n_comp:6,d} of the {int(compact.sum()):,} compact frames "
          f"({n_comp/max(compact.sum(),1):.2e})")

print("\nThe conjunction that cannot be met by chance:")
strict = (r <= 1.0) & (q >= 0.8) & (yw <= 6.0)
mid = (r <= 1.5) & (q >= 0.7) & (yw <= 7.0)
print(f"  RMSD <= 1.0 A AND Q >= 0.80 AND Tyr2/Trp9 <= 6.0 A : {int(strict.sum()):,} frames")
print(f"  RMSD <= 1.5 A AND Q >= 0.70 AND Tyr2/Trp9 <= 7.0 A : {int(mid.sum()):,} frames")
if strict.sum():
    i = np.where(strict)[0][np.argmin(r[strict])]
    print(f"  best such frame: RMSD {r[i]:.2f} A, Q {q[i]:.3f}, Tyr2/Trp9 {yw[i]:.2f} A, CV1 {cv[i]:.3f}")
print("\n  A structure that is compact by accident reproduces neither 1UAO's specific contact")
print("  map (Q) nor its hydrophobic cluster. The conjunction is the evidence; the RMSD count")
print("  alone is not, because the matched null gives it a 25% baseline.")

json.dump({"q_max": float(q.max()), "n_q_ge_0.8": int((q >= 0.8).sum()),
           "n_q_ge_0.7": int((q >= 0.7).sum()),
           "n_strict_conjunction": int(strict.sum()), "n_mid_conjunction": int(mid.sum()),
           "q_median_compact": float(np.median(q[compact])),
           "q_p99_compact": float(np.percentile(q[compact], 99))},
          open(f"{OUT}/q_discrimination.json", "w"), indent=2)
