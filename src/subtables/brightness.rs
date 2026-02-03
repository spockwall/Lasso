use super::SubtableStrategy;
use crate::subtables::EqPolynomial;
use crate::utils::split_bits;
use ark_ff::PrimeField;
use ark_std::log2;

///
/// - Formula of brightness adjustment: Y' = clip(Y + b)
/// - b > 0: Brighter
/// - b < 0: Darker
///
/// - Input: Y (0-255), b (-128 to 127)
/// - Output: Y' (0-255)
///

pub enum BrightnessSubtableStrategy {}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
  for BrightnessSubtableStrategy
{
  const NUM_SUBTABLES: usize = 1;
  const NUM_MEMORIES: usize = C;

  /// Index = (Y || b), compute Y' = clip(Y + b)
  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    let mut materialized: Vec<F> = Vec::with_capacity(M);
    // assume M=2^16 (65536), bits_per_operand = 8
    let bits_per_operand = (log2(M) / 2) as usize;

    for idx in 0..M {
      // split index into high bits (lhs/Y) and low bits (rhs/b)
      let (y_raw, b_raw) = split_bits(idx, bits_per_operand);

      // y is unsigned pixel (0-255)
      let y = y_raw as i16;
      // b is signed adjustment value (we assume raw u8 -> i8 via 2's complement)
      let b = (b_raw as u8) as i8 as i16;

      // core logic: add and clip
      let result = y + b;
      let val = result.clamp(0, 255);

      materialized.push(F::from(val as u64));
    }

    std::array::from_fn(|_| materialized.clone())
  }

  /// Verifier verifies (MLE Evaluation) using EqPolynomial
  /// points include log2(M) random variables, i.e., M = 2^16
  fn evaluate_subtable_mle(_: usize, point: &Vec<F>) -> F {
    // initialize EqPolynomial
    let eq_poly = EqPolynomial::new(point.clone());

    // use Tensor Product optimization (compute_factored_evals)
    // this will automatically split point into two halves, and generate two small tables L and R
    // L corresponds to the first half of point (Low bits / b)
    // R corresponds to the second half of point (High bits / y)
    let (eq_b_table, eq_y_table) = eq_poly.compute_factored_evals();

    // ensure dimension is correct (2^8 = 256)
    let limit = eq_b_table.len();
    // or: let limit = 1 << (point.len() / 2);

    let mut sum = F::zero();

    // double loop scan
    // Notice: we now use eq_y_table to traverse y, eq_b_table to traverse b
    for y_idx in 0..limit {
      let weight_y = eq_y_table[y_idx];
      if weight_y == F::zero() {
        continue;
      }

      for b_idx in 0..limit {
        let weight_b = eq_b_table[b_idx];

        // real logic: Y' = clamp(y + b)
        // notice: type conversion must match materialize
        let y = y_idx as i16;
        let b = (b_idx as u8) as i8 as i16;

        let res = y + b;
        let val = res.clamp(0, 255);

        // accumulate: Value * Weight_Y * Weight_b
        sum += F::from(val as u64) * weight_y * weight_b;
      }
    }

    sum
  }

  /// If C > 1, it means we are processing C pixels at the same time (SIMD)
  /// The result will be combined into a large integer: Res[0] + 2^16*Res[1] ...
  /// T = T'[0] + 2^16*T'[1] + 2^32*T'[2] + 2^48*T'[3]
  /// T'[3] | T'[2] | T'[1] | T'[0]
  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
    let increment = log2(M) as usize; // e.g., 16 bits
    let mut sum = F::zero();
    for i in 0..C {
      // shift 0, 16, 32, 48 ...
      let weight: u64 = 1u64 << (i * increment);
      sum += F::from(weight) * vals[i];
    }
    sum
  }

  fn g_poly_degree() -> usize {
    1
  }
}

#[cfg(test)]
mod test {
  use crate::{
    materialization_mle_parity_test, subtables::Subtables, utils::index_to_field_bitvector,
  };

  use super::*;
  use ark_curve25519::Fr;

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 4;
    const M: usize = 1 << 4; // 16 entries, 4 bits total (2 bits for y, 2 bits for b)

    let materialized: [Vec<Fr>; 1] =
      <BrightnessSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].len(), M);

    let table: Vec<Fr> = materialized[0].clone();

    // Logic: y + b
    // M=16 -> 4 bits total -> 2 bits per operand.
    // y range: 0..3, b range: 0..3 (b is positive because 2-bit u8 cast to i8 is positive)

    // Index 0: y=0 (00), b=0 (00) -> 0
    assert_eq!(table[0], Fr::from(0));
    // Index 1: y=0 (00), b=1 (01) -> 1
    assert_eq!(table[1], Fr::from(1));
    // Index 2: y=0 (00), b=2 (10) -> 2
    assert_eq!(table[2], Fr::from(2));
    // Index 3: y=0 (00), b=3 (11) -> 3
    assert_eq!(table[3], Fr::from(3));

    // Index 4: y=1 (01), b=0 (00) -> 1
    assert_eq!(table[4], Fr::from(1));
    // Index 5: y=1 (01), b=1 (01) -> 2
    assert_eq!(table[5], Fr::from(2));
    // Index 6: y=1 (01), b=2 (10) -> 3
    assert_eq!(table[6], Fr::from(3));
    // Index 7: y=1 (01), b=3 (11) -> 4
    assert_eq!(table[7], Fr::from(4));

    // Index 8: y=2 (10), b=0 (00) -> 2
    assert_eq!(table[8], Fr::from(2));
    // Index 10: y=2 (10), b=2 (10) -> 4
    assert_eq!(table[10], Fr::from(4));

    // Index 15: y=3 (11), b=3 (11) -> 6
    assert_eq!(table[15], Fr::from(6));
  }

  #[test]
  fn combine() {
    const M: usize = 1 << 16;
    let combined: Fr =
      <BrightnessSubtableStrategy as SubtableStrategy<Fr, 4, M>>::combine_lookups(&[
        Fr::from(100),
        Fr::from(200),
        Fr::from(300),
        Fr::from(400),
      ]);

    // Standard Jolt combination: val_0 + 2^16*val_1 + 2^32*val_2 + ...
    let expected = (1u64 * 100u64)
      + ((1u64 << 16u64) * 200u64)
      + ((1u64 << 32u64) * 300u64)
      + ((1u64 << 48u64) * 400u64);
    assert_eq!(combined, Fr::from(expected));
  }

  #[test]
  fn valid_merged_poly() {
    const C: usize = 2;
    const M: usize = 1 << 4;

    // Simulate two lookups (C=2)
    let x_indices: Vec<usize> = vec![0, 2]; // Lookup indices for Memory 0
    let y_indices: Vec<usize> = vec![5, 9]; // Lookup indices for Memory 1

    let subtable_evals: Subtables<Fr, C, M, BrightnessSubtableStrategy> =
      Subtables::new(&[x_indices, y_indices], 2);

    // combined_table_index_bits = log2(sparsity) + log2(C) = 1 + 1 = 2
    let combined_table_index_bits = 2;

    for (x, expected) in vec![
      // Lookup 0 (from Memory 0): Index 0 -> y=0, b=0 -> Res=0
      (0, 0),
      // Lookup 1 (from Memory 0): Index 2 -> y=0, b=2 -> Res=2
      (1, 2),
      // Lookup 2 (from Memory 1): Index 5 -> y=1, b=1 -> Res=2
      (2, 2),
      // Lookup 3 (from Memory 1): Index 9 -> y=2, b=1 -> Res=3
      (3, 3),
    ] {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, combined_table_index_bits));
      assert_eq!(calculated, Fr::from(expected));
    }
  }

  // Automatically verifies that evaluate_mle(r) equals evaluate_poly(materialize(), r)
  // This ensures the optimized "tensor product" MLE evaluation in the Verifier matches
  // the naive table materialization in the Prover.
  materialization_mle_parity_test!(
    materialization_parity,
    BrightnessSubtableStrategy,
    Fr,
    16, // M = 2^16
    1   // C = 1
  );

  materialization_mle_parity_test!(
    materialization_parity_nonzero_c,
    BrightnessSubtableStrategy,
    Fr,
    16, // M = 2^16
    2   // C = 2
  );
}
