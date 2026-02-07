use crate::lasso::{
  densified::DensifiedRepresentation,
  surge::{SparseLookupMatrix, SparsePolyCommitmentGens, SparsePolynomialEvaluationProof},
};
use crate::subtables::contrast::ContrastSubtableStrategy;
use crate::utils::random::RandomTape;
use ark_curve25519::{EdwardsProjective, Fr};
use ark_std::log2;
use merlin::Transcript;

pub fn contrast_benchmarks() -> Vec<(tracing::Span, fn())> {
  vec![
    // Single pixel benchmarks (C=1)
    crate::single_pass_lasso!(
      "Contrast(2^10)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 10
    ),
    crate::single_pass_lasso!(
      "Contrast(2^12)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 12
    ),
    crate::single_pass_lasso!(
      "Contrast(2^14)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    crate::single_pass_lasso!(
      "Contrast(2^16)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    crate::single_pass_lasso!(
      "Contrast(2^18)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    crate::single_pass_lasso!(
      "Contrast(2^20)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
    crate::single_pass_lasso!(
      "Contrast(2^22)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 22
    ),
    // Batch processing benchmarks (C=4, processing 4 pixels simultaneously)
    crate::single_pass_lasso!(
      "Contrast_4x(2^14)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    crate::single_pass_lasso!(
      "Contrast_4x(2^16)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    crate::single_pass_lasso!(
      "Contrast_4x(2^18)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    // Batch processing benchmarks (C=8, processing 8 pixels simultaneously)
    crate::single_pass_lasso!(
      "Contrast_8x(2^16)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    crate::single_pass_lasso!(
      "Contrast_8x(2^18)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    crate::single_pass_lasso!(
      "Contrast_8x(2^20)",
      Fr,
      EdwardsProjective,
      ContrastSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
  ]
}
