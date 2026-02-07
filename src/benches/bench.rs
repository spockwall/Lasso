use crate::lasso::surge::SparsePolyCommitmentGens;
use crate::subtables::and::AndSubtableStrategy;
use crate::subtables::brightness::BrightnessSubtableStrategy;
use crate::{
  lasso::{
    densified::DensifiedRepresentation,
    surge::{SparseLookupMatrix, SparsePolynomialEvaluationProof},
  },
  utils::random::RandomTape,
};
use ark_curve25519::{EdwardsProjective, Fr};
use ark_ff::PrimeField;
use ark_std::{log2, test_rng};
use merlin::Transcript;
use rand_chacha::rand_core::RngCore;

pub fn gen_indices<const C: usize>(sparsity: usize, memory_size: usize) -> Vec<[usize; C]> {
  let mut rng = test_rng();
  let mut all_indices: Vec<[usize; C]> = Vec::new();
  for _ in 0..sparsity {
    let indices = [rng.next_u64() as usize % memory_size; C];
    all_indices.push(indices);
  }
  all_indices
}

pub fn gen_random_points<F: PrimeField, const C: usize>(memory_bits: usize) -> [Vec<F>; C] {
  std::array::from_fn(|_| gen_random_point(memory_bits))
}

pub fn gen_random_point<F: PrimeField>(memory_bits: usize) -> Vec<F> {
  let mut rng = test_rng();
  let mut r_i: Vec<F> = Vec::with_capacity(memory_bits);
  for _ in 0..memory_bits {
    r_i.push(F::rand(&mut rng));
  }
  r_i
}

macro_rules! single_pass_lasso {
  ($span_name:expr, $field:ty, $group:ty, $subtable_strategy:ty, $C:expr, $M:expr, $sparsity:expr) => {
    (tracing::info_span!($span_name), move || {
      const C: usize = $C;
      const M: usize = $M;
      const S: usize = $sparsity;
      type F = $field;
      type G = $group;
      type SubtableStrategy = $subtable_strategy;

      let log_m = log2(M) as usize;
      let log_s: usize = log2($sparsity) as usize;

      let r: Vec<F> = gen_random_point::<F>(log_s);

      let nz = gen_indices::<C>(S, M);
      let lookup_matrix = SparseLookupMatrix::new(nz.clone(), log_m);

      // Prove
      let mut dense: DensifiedRepresentation<F, C> = DensifiedRepresentation::from(&lookup_matrix);
      let gens = SparsePolyCommitmentGens::<G>::new(b"gens_sparse_poly", C, S, C, log_m);
      let _commitment = dense.commit::<$group>(&gens);
      let mut random_tape = RandomTape::new(b"proof");
      let mut prover_transcript = Transcript::new(b"example");
      let proof = SparsePolynomialEvaluationProof::<G, C, M, SubtableStrategy>::prove(
        &mut dense,
        &r,
        &gens,
        &mut prover_transcript,
        &mut random_tape,
      );

      // Calculate proof size
      use ark_serialize::CanonicalSerialize;
      let mut proof_bytes = Vec::new();
      proof
        .serialize_compressed(&mut proof_bytes)
        .expect("Serialization failed");
      let proof_size = proof_bytes.len();

      tracing::info!(
        proof_size_bytes = proof_size,
        proof_size_kb = format!("{:.2}", proof_size as f64 / 1024.0),
        sparsity = S,
        memory_size = M,
        columns = C,
        "Proof size"
      );
    })
  };
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum BenchType {
  JoltDemo,
  Halo2Comparison,
  Brightness,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, fn())> {
  match bench_type {
    BenchType::JoltDemo => jolt_demo_benchmarks(),
    BenchType::Halo2Comparison => halo2_comparison_benchmarks(),
    BenchType::Brightness => brightness_benchmarks(),
    _ => panic!("BenchType does not have a mapping"),
  }
}

fn jolt_demo_benchmarks() -> Vec<(tracing::Span, fn())> {
  vec![
    single_pass_lasso!(
      "And(2^128, 2^10)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 10
    ),
    single_pass_lasso!(
      "And(2^128, 2^12)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 12
    ),
    single_pass_lasso!(
      "And(2^128, 2^14)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    single_pass_lasso!(
      "And(2^128, 2^16)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "And(2^128, 2^18)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    single_pass_lasso!(
      "And(2^128, 2^20)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
    single_pass_lasso!(
      "And(2^128, 2^22)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 22
    ),
  ]
}

fn halo2_comparison_benchmarks() -> Vec<(tracing::Span, fn())> {
  vec![
    single_pass_lasso!(
      "And(2^10)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 10
    ),
    single_pass_lasso!(
      "And(2^12)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 12
    ),
    single_pass_lasso!(
      "And(2^14)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    single_pass_lasso!(
      "And(2^16)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "And(2^18)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    single_pass_lasso!(
      "And(2^20)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
    single_pass_lasso!(
      "And(2^22)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 22
    ),
    single_pass_lasso!(
      "And(2^24)",
      Fr,
      EdwardsProjective,
      AndSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 24
    ),
  ]
}

fn brightness_benchmarks() -> Vec<(tracing::Span, fn())> {
  vec![
    // Single pixel benchmarks (C=1)
    single_pass_lasso!(
      "Brightness(2^10)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 10
    ),
    single_pass_lasso!(
      "Brightness(2^12)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 12
    ),
    single_pass_lasso!(
      "Brightness(2^14)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    single_pass_lasso!(
      "Brightness(2^16)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "Brightness(2^18)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    single_pass_lasso!(
      "Brightness(2^20)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
    single_pass_lasso!(
      "Brightness(2^22)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 1,
      /* M= */ 1 << 16,
      /* S= */ 1 << 22
    ),
    // Batch processing benchmarks (C=4, processing 4 pixels simultaneously)
    single_pass_lasso!(
      "Brightness_4x(2^14)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 14
    ),
    single_pass_lasso!(
      "Brightness_4x(2^16)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "Brightness_4x(2^18)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 4,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    // Batch processing benchmarks (C=8, processing 8 pixels simultaneously)
    single_pass_lasso!(
      "Brightness_8x(2^16)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 16
    ),
    single_pass_lasso!(
      "Brightness_8x(2^18)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 18
    ),
    single_pass_lasso!(
      "Brightness_8x(2^20)",
      Fr,
      EdwardsProjective,
      BrightnessSubtableStrategy,
      /* C= */ 8,
      /* M= */ 1 << 16,
      /* S= */ 1 << 20
    ),
  ]
}
