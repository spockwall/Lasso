use super::SubtableStrategy;
use crate::subtables::EqPolynomial;
use crate::utils::split_bits;
use ark_ff::PrimeField;
use ark_std::log2;

///
/// Shadow Adjustment Subtable Strategy
///
/// - Formula: Y' = clip(Y + s * w(Y)) if Y < t else Y
/// - Parameters:
///   - s: shadow strength (e.g. 0.3)
///   - p: shadow power (e.g. 2)
///   - t: shadow cutoff (e.g. 0.3)
/// - Weighting function:
///   - w(Y) = 1 - (Y/t)^p
///
/// - Adjusts brightness of dark pixels (Y < t) by scaling them with w(Y)
///
/// - Input: Y (0-255), s (0-255)
///   - Y: pixel brightness value
///   - s: shadow strength
///   - t: shadow cutoff, fixed to 0.3
///   - p: shadow power, fixed to 2
/// - Output: Y' (0-255)
///
///
///
/// ### 3. Shadow

pub enum ShadowSubtableStrategy {}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
  for ShadowSubtableStrategy
{
  const NUM_SUBTABLES: usize = 1;
  const NUM_MEMORIES: usize = C;

  /// Index = (Y || s), compute Y' = clip(Y + s * w(Y))
  ///
  /// s is interpreted as i8 (-128 to 127), representing shadow strength
  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    let mut materialized: Vec<F> = Vec::with_capacity(M);
    // assume M=2^16 (65536), bits_per_operand = 8
    let bits_per_operand = (log2(M) / 2) as usize;

    const THRESHOLD: f64 = 77.0;
    const POWER: f64 = 2.0;

    for idx in 0..M {
      // split index into high bits (lhs/Y) and low bits (rhs/c)
      let (y_raw, s_raw) = split_bits(idx, bits_per_operand);

      // y is unsigned pixel (0-255)
      let y = y_raw as f64;
      // s is shadow strength (raw u8 -> i8 via 2's complement)
      let s = (s_raw as u8) as i8 as f64;

      // core logic: shadow adjustment around midpoint
      let result = if y >= THRESHOLD {
        y
      } else {
        // calculate weighting: w(Y)
        let ratio = y / THRESHOLD;
        let weight = 1.0 - ratio.powf(POWER);

        // calculate brightness boost
        y + s * weight
      };
      let val = result.clamp(0.0, 255.0);

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
    // L corresponds to the first half of point (High bits / y)
    // R corresponds to the second half of point (Low bits / s)
    let (eq_y_table, eq_s_table) = eq_poly.compute_factored_evals();

    // ensure dimension is correct (2^8 = 256)
    let limit = eq_s_table.len();
    // or: let limit = 1 << (point.len() / 2);

    let mut sum = F::zero();

    const THRESHOLD: f64 = 77.0;
    const POWER: f64 = 2.0;

    // double loop scan
    // Notice: we now use eq_y_table to traverse y, eq_s_table to traverse s
    for y_idx in 0..limit {
      let weight_y = eq_y_table[y_idx];
      if weight_y == F::zero() {
        continue;
      }

      let y = y_idx as f64;
      let w_factor = if y >= THRESHOLD {
        0.0
      } else {
        let ratio = y / THRESHOLD;
        1.0 - ratio.powf(POWER)
      };

      for s_idx in 0..limit {
        let weight_s = eq_s_table[s_idx];

        // real logic: Y' = clip(y + s * w(y))
        // notice: type conversion must match materialize
        let s = (s_idx as u8) as i8 as f64;

        let res = y + s * w_factor;
        let val = res.clamp(0.0, 255.0);

        // accumulate: Value * Weight_Y * Weight_c
        sum += F::from(val as u64) * weight_y * weight_s;
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
    materialization_mle_parity_test,
    subtables::{shadow::ShadowSubtableStrategy, Subtables},
    utils::index_to_field_bitvector,
  };

  use super::*;
  use ark_curve25519::Fr;

  /// Helper function to calculate shadow adjustment
  /// Formula: Y' = clip(Y + s * w(Y)) if Y < t else Y
  /// where w(Y) = 1 - (Y/t)^p
  fn shadow_adjust(y: u8, s: i8) -> u8 {
    const THRESHOLD: f64 = 77.0;
    const POWER: f64 = 2.0;

    let y_f64 = y as f64;
    let s_f64 = s as f64;

    let result = if y_f64 >= THRESHOLD {
      y_f64
    } else {
      let ratio = y_f64 / THRESHOLD;
      let weight = 1.0 - ratio.powf(POWER);
      y_f64 + s_f64 * weight
    };

    result.clamp(0.0, 255.0) as u8
  }

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 4;
    const M: usize = 1 << 4; // 16 entries, 4 bits total (2 bits for y, 2 bits for s)

    let materialized: [Vec<Fr>; 1] =
      <ShadowSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].len(), M);

    let table: Vec<Fr> = materialized[0].clone();

    // Shadow formula: Y' = clip(Y + s * w(Y)) if Y < t else Y
    // where w(Y) = 1 - (Y/t)^2, t = 77
    // M=16 -> 4 bits total -> 2 bits per operand
    // y range: 0..3, s range: 0..3 (as i8: 0, 1, 2, 3 are all positive)
    // All y values (0-3) are below threshold (77), so shadow adjustment applies

    // Index 0: y=0, s=0 -> w(0) = 1 - 0 = 1, result = 0 + 0*1 = 0
    assert_eq!(table[0], Fr::from(0));

    // Index 1: y=0, s=1 -> w(0) = 1, result = 0 + 1*1 = 1
    assert_eq!(table[1], Fr::from(1));

    // Index 5: y=1, s=1 -> w(1) = 1 - (1/77)^2 ≈ 0.9998, result ≈ 1 + 1*0.9998 ≈ 1 (truncated)
    assert_eq!(table[5], Fr::from(1));

    // Index 15: y=3, s=3 -> w(3) = 1 - (3/77)^2 ≈ 0.9985, result ≈ 3 + 3*0.9985 ≈ 5
    assert_eq!(table[15], Fr::from(5));
  }

  #[test]
  fn test_shadow_edge_cases() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ShadowSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    const THRESHOLD: u8 = 77;

    // Test pixels above threshold: should remain unchanged regardless of s
    for s_raw in 0..=255u8 {
      let s = s_raw as i8;
      for y in THRESHOLD..=255u8 {
        let idx = (y as usize) << 8 | s_raw as usize;
        let expected = shadow_adjust(y, s);
        assert_eq!(
          table[idx],
          Fr::from(expected as u64),
          "Failed for Y={}, s={}",
          y,
          s
        );
        assert_eq!(expected, y, "Pixels above threshold should be unchanged");
      }
    }

    // Test s=0: all pixels should remain unchanged
    for y in 0..=255u8 {
      let idx = (y as usize) << 8; // s=0
      let expected = shadow_adjust(y, 0);
      assert_eq!(table[idx], Fr::from(expected as u64));
      assert_eq!(expected, y, "s=0 should leave all pixels unchanged");
    }

    // Test maximum positive shadow strength (s=127): maximum brightening for dark pixels
    let test_cases: Vec<(usize, u8)> = vec![
      (0, 127),   // Darkest pixel with max shadow lift: 0 + 127*1 = 127
      (30, 137),  // Dark pixel: 30 + 127*w(30) ≈ 137
      (50, 123),  // Mid-dark pixel: 50 + 127*w(50) ≈ 123
      (76, 79),   // Just below threshold: 76 + 127*w(76) ≈ 79
      (77, 77),   // At threshold, no change
      (128, 128), // Above threshold, no change
      (255, 255), // Brightest pixel, no change
    ];

    for (y, expected) in test_cases {
      let idx = (y << 8) | 127;
      let calculated = shadow_adjust(y as u8, 127);
      assert_eq!(table[idx], Fr::from(calculated as u64));
      assert_eq!(calculated, expected, "Failed for Y={}, s=127", y);
    }
  }

  #[test]
  fn test_shadow_clipping() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ShadowSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    // Test that all values are properly clipped to [0, 255]
    for idx in 0..M {
      let value = table[idx];
      // Convert Fr back to u64 for comparison
      let value_u64 = value.into_bigint().0[0];
      assert!(
        value_u64 <= 255,
        "Value at index {idx} exceeds 255: {value_u64}"
      );
    }

    // Test specific clipping cases with high shadow strength
    let high_shadow_cases: Vec<(usize, usize, u8)> = vec![
      (0, 127, 127),  // Darkest pixel with max shadow: 0 + 127*1 = 127
      (10, 127, 134), // Dark pixel: 10 + 127*w(10) ≈ 134
      (50, 120, 119), // Mid-dark pixel: 50 + 120*w(50) ≈ 119
      (76, 127, 79),  // Just below threshold: 76 + 127*w(76) ≈ 79
    ];

    for (y, s_raw, expected) in high_shadow_cases {
      let s = s_raw as i8;
      let idx = (y << 8) | s_raw;
      let calculated = shadow_adjust(y as u8, s);
      assert_eq!(table[idx], Fr::from(calculated as u64));
      assert_eq!(calculated, expected, "Clipping failed for Y={y}, s={s}");
    }
  }

  #[test]
  fn combine() {
    const M: usize = 1 << 16;
    let combined: Fr = <ShadowSubtableStrategy as SubtableStrategy<Fr, 4, M>>::combine_lookups(&[
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
    const M: usize = 1 << 16;

    // Use realistic indices for M=2^16
    // Index format: (Y << 8) | c
    let x_indices: Vec<usize> = vec![
      (100 << 8) | 64, // Y=100, c=64
      (200 << 8) | 96, // Y=200, c=96
    ];
    let y_indices: Vec<usize> = vec![
      (128 << 8) | 0,  // Y=128, c=0 -> always 128
      (50 << 8) | 127, // Y=50, c=127 -> high contrast on dark pixel
    ];

    let subtable_evals: Subtables<Fr, C, M, ShadowSubtableStrategy> =
      Subtables::new(&[x_indices.clone(), y_indices.clone()], 2);

    let combined_table_index_bits = 2; // log2(sparsity=2) + log2(C=2) = 1 + 1 = 2

    // Calculate expected values
    let expected_values = vec![
      shadow_adjust(100, 64), // Lookup 0: Y=100, c=64
      shadow_adjust(200, 96), // Lookup 1: Y=200, c=96
      shadow_adjust(128, 0),  // Lookup 2: Y=128, c=0
      shadow_adjust(50, 127), // Lookup 3: Y=50, c=127
    ];

    for (x, expected) in expected_values.iter().enumerate() {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, combined_table_index_bits));
      assert_eq!(
        calculated,
        Fr::from(*expected as u64),
        "Failed for lookup {x}: expected {expected}, got {calculated:?}"
      );
    }
  }

  // Automatically verifies that evaluate_mle(r) equals evaluate_poly(materialize(), r)
  // This ensures the optimized "tensor product" MLE evaluation in the Verifier matches
  // the naive table materialization in the Prover.
  materialization_mle_parity_test!(
    materialization_parity,
    ShadowSubtableStrategy,
    Fr,
    1 << 16, // M = 2^16 = 65536
    1        // C = 1
  );

  materialization_mle_parity_test!(
    materialization_parity_nonzero_c,
    ShadowSubtableStrategy,
    Fr,
    1 << 16, // M = 2^16 = 65536
    2        // C = 2
  );
}
