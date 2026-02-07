use super::SubtableStrategy;
use crate::subtables::EqPolynomial;
use crate::utils::split_bits;
use ark_ff::PrimeField;
use ark_std::log2;

///
/// Contrast Adjustment Subtable Strategy
///
/// - Formula: Y' = clip((Y - 128) * c / 128 + 128)
/// - Adjusts contrast by scaling pixel values around the midpoint (128)
///
/// - Input: Y (0-255), c (-128 to 127)
///   - Y: pixel brightness value
///   - c: contrast factor (interpreted as i8)
///     - c = 0: minimum contrast (all pixels → 128)
///     - c = 127: maximum contrast (2x scaling)
///     - c = -128: inverted minimum contrast
/// - Output: Y' (0-255)
///

pub enum ContrastSubtableStrategy {}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
  for ContrastSubtableStrategy
{
  const NUM_SUBTABLES: usize = 1;
  const NUM_MEMORIES: usize = C;

  /// Index = (Y || c), compute Y' = clip((Y - 128) * c / 128 + 128)
  ///
  /// c is interpreted as i8 (-128 to 127), representing contrast scaling factor
  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    let mut materialized: Vec<F> = Vec::with_capacity(M);
    // assume M=2^16 (65536), bits_per_operand = 8
    let bits_per_operand = (log2(M) / 2) as usize;

    let mid_gray = 128;

    for idx in 0..M {
      // split index into high bits (lhs/Y) and low bits (rhs/c)
      let (y_raw, c_raw) = split_bits(idx, bits_per_operand);

      // y is unsigned pixel (0-255)
      let y = y_raw as i16;
      // c is contrast factor (raw u8 -> i8 via 2's complement)
      let c = (c_raw as u8) as i8 as i16;

      // core logic: contrast adjustment around midpoint
      let result = (y - mid_gray) * c / mid_gray + mid_gray;
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
    // L corresponds to the first half of point (High bits / y)
    // R corresponds to the second half of point (Low bits / c)
    let (eq_y_table, eq_c_table) = eq_poly.compute_factored_evals();

    // ensure dimension is correct (2^8 = 256)
    let limit = eq_c_table.len();
    // or: let limit = 1 << (point.len() / 2);

    let mut sum = F::zero();

    // double loop scan
    // Notice: we now use eq_y_table to traverse y, eq_c_table to traverse c
    for y_idx in 0..limit {
      let weight_y = eq_y_table[y_idx];
      if weight_y == F::zero() {
        continue;
      }

      for c_idx in 0..limit {
        let weight_c = eq_c_table[c_idx];

        // real logic: Y' = clamp((y - 128) * c / 128 + 128)
        // notice: type conversion must match materialize
        let y = y_idx as i16;
        let c = (c_idx as u8) as i8 as i16;

        let res = (y - 128) * c / 128 + 128;
        let val = res.clamp(0, 255);

        // accumulate: Value * Weight_Y * Weight_c
        sum += F::from(val as u64) * weight_y * weight_c;
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
    subtables::{contrast::ContrastSubtableStrategy, Subtables},
    utils::index_to_field_bitvector,
  };

  use super::*;
  use ark_curve25519::Fr;

  /// Helper function to calculate contrast adjustment
  fn contrast_adjust(y: i16, c: i16) -> i16 {
    let mid_gray = 128;
    let result = (y - mid_gray) * c / mid_gray + mid_gray;
    result.clamp(0, 255)
  }

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 4;
    const M: usize = 1 << 4; // 16 entries, 4 bits total (2 bits for y, 2 bits for c)

    let materialized: [Vec<Fr>; 1] =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].len(), M);

    let table: Vec<Fr> = materialized[0].clone();

    // Contrast formula: Y' = clip((Y - 128) * c / 128 + 128)
    // M=16 -> 4 bits total -> 2 bits per operand
    // y range: 0..3, c range: 0..3 (c as i8: 0, 1, 2, 3 are all positive)
    // Note: With only 2 bits, all values are small and close to midpoint

    // For small M, the division by 128 makes most results = 128 (midpoint)
    // Let's verify a few key indices:

    // Index 0: y=0, c=0 -> (0-128)*0/128+128 = 128
    assert_eq!(table[0], Fr::from(128));

    // Index 1: y=0, c=1 -> (0-128)*1/128+128 = -128/128+128 = 127
    assert_eq!(table[1], Fr::from(127));

    // Index 5: y=1, c=1 -> (1-128)*1/128+128 = -127/128+128 = 128 (integer division)
    assert_eq!(table[5], Fr::from(128));

    // Index 15: y=3, c=3 -> (3-128)*3/128+128 = -375/128+128 = 126 (integer division: -375/128 = -2)
    assert_eq!(table[15], Fr::from(126));
  }

  #[test]
  fn test_contrast_edge_cases() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    // Test midpoint (Y=128) with various contrast values
    // At midpoint, result should always be 128 regardless of c
    for c_raw in 0..=255u8 {
      let c = c_raw as i8 as i16;
      let idx = (128 << 8) | c_raw as usize;
      let expected = contrast_adjust(128, c);
      assert_eq!(
        table[idx],
        Fr::from(expected as u64),
        "Failed for Y=128, c={}",
        c
      );
      assert_eq!(expected, 128, "Midpoint should always be 128");
    }

    // Test minimum contrast (c=0): all pixels should become 128
    for y in 0..=255u8 {
      let idx = (y as usize) << 8; // c=0
      let expected = contrast_adjust(y as i16, 0);
      assert_eq!(table[idx], Fr::from(expected as u64));
      assert_eq!(expected, 128, "c=0 should map all pixels to 128");
    }

    // Test maximum positive contrast (c=127): maximum scaling
    let test_cases = vec![
      (0, 1),     // (0-128)*127/128+128 = -16256/128+128 = -127+128 = 1
      (64, 65),   // (64-128)*127/128+128 = -64*127/128+128 = -8128/128+128 = -63+128 = 65
      (128, 128), // Midpoint unchanged
      (192, 191), // (192-128)*127/128+128 = 64*127/128+128 = 8128/128+128 = 63+128 = 191
      (255, 254), // (255-128)*127/128+128 = 127*127/128+128 = 16129/128+128 = 126+128 = 254
    ];

    for (y, expected) in test_cases {
      let idx = (y << 8) | 127;
      let calculated = contrast_adjust(y as i16, 127);
      assert_eq!(table[idx], Fr::from(calculated as u64));
      assert_eq!(calculated, expected, "Failed for Y={}, c=127", y);
    }
  }

  #[test]
  fn test_contrast_formula_correctness() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    // Test specific known values
    let test_cases = vec![
      // (Y, c, expected_Y')
      (128, 64, 128),  // Midpoint with any c -> 128
      (200, 64, 164),  // (200-128)*64/128+128 = 72*64/128+128 = 4608/128+128 = 36+128 = 164
      (50, 64, 89),    // (50-128)*64/128+128 = -78*64/128+128 = -4992/128+128 = -39+128 = 89
      (255, 127, 254), // (255-128)*127/128+128 = 127*127/128+128 = 16129/128+128 = 126+128 = 254
      (0, 127, 1),     // (0-128)*127/128+128 = -128*127/128+128 = -16256/128+128 = -127+128 = 1
      (180, 96, 167),  // (180-128)*96/128+128 = 52*96/128+128 = 4992/128+128 = 39+128 = 167
    ];

    for (y, c_raw, expected) in test_cases {
      let c = c_raw as i8 as i16;
      let idx = (y << 8) | c_raw;
      let calculated = contrast_adjust(y as i16, c);
      assert_eq!(table[idx], Fr::from(calculated as u64));
      assert_eq!(
        calculated, expected,
        "Failed for Y={}, c={}: expected {}, got {}",
        y, c, expected, calculated
      );
    }
  }

  #[test]
  fn test_contrast_symmetry() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    // Test that contrast adjustment is symmetric around midpoint
    // For a given c, if Y1 is below midpoint by d, and Y2 is above by d,
    // then the adjustments should be symmetric
    let c_values = vec![32i8, 64, 96, 127];

    for c_raw in c_values {
      let c = c_raw as i16;
      for offset in 1..=64 {
        let y1 = 128 - offset;
        let y2 = 128 + offset;

        let idx1 = (y1 << 8) | (c_raw as u8) as usize;
        let idx2 = (y2 << 8) | (c_raw as u8) as usize;

        let result1 = contrast_adjust(y1 as i16, c);
        let result2 = contrast_adjust(y2 as i16, c);

        assert_eq!(table[idx1], Fr::from(result1 as u64));
        assert_eq!(table[idx2], Fr::from(result2 as u64));

        // Check symmetry: distance from midpoint should be proportional
        let dist1 = (128 - result1).abs();
        let dist2 = (result2 - 128).abs();

        // Due to integer division, allow small difference
        assert!(
          (dist1 - dist2).abs() <= 1,
          "Asymmetry detected for c={}, offset={}: dist1={}, dist2={}",
          c,
          offset,
          dist1,
          dist2
        );
      }
    }
  }

  #[test]
  fn test_contrast_clipping() {
    const C: usize = 1;
    const M: usize = 1 << 16;

    let materialized: [Vec<Fr>; 1] =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let table = &materialized[0];

    // Test that all values are properly clipped to [0, 255]
    for idx in 0..M {
      let value = table[idx];
      // Convert Fr back to u64 for comparison
      let value_u64 = value.into_bigint().0[0];
      assert!(
        value_u64 <= 255,
        "Value at index {} exceeds 255: {}",
        idx,
        value_u64
      );
    }

    // Test specific clipping cases with high contrast
    let high_contrast_cases = vec![
      (0, 127, 1),     // (0-128)*127/128+128 = -127+128 = 1 (not clipped to 0)
      (10, 127, 11),   // (10-128)*127/128+128 = -118*127/128+128 = -14986/128+128 = -117+128 = 11
      (245, 127, 244), // (245-128)*127/128+128 = 117*127/128+128 = 14859/128+128 = 116+128 = 244
      (255, 127, 254), // (255-128)*127/128+128 = 127*127/128+128 = 126+128 = 254
    ];

    for (y, c_raw, expected) in high_contrast_cases {
      let c = c_raw as i8 as i16;
      let idx = (y << 8) | c_raw;
      let calculated = contrast_adjust(y as i16, c);
      assert_eq!(table[idx], Fr::from(calculated as u64));
      assert_eq!(calculated, expected, "Clipping failed for Y={}, c={}", y, c);
    }
  }

  #[test]
  fn combine() {
    const M: usize = 1 << 16;
    let combined: Fr =
      <ContrastSubtableStrategy as SubtableStrategy<Fr, 4, M>>::combine_lookups(&[
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

    let subtable_evals: Subtables<Fr, C, M, ContrastSubtableStrategy> =
      Subtables::new(&[x_indices.clone(), y_indices.clone()], 2);

    let combined_table_index_bits = 2; // log2(sparsity=2) + log2(C=2) = 1 + 1 = 2

    // Calculate expected values
    let expected_values = vec![
      contrast_adjust(100, 64), // Lookup 0: Y=100, c=64
      contrast_adjust(200, 96), // Lookup 1: Y=200, c=96
      contrast_adjust(128, 0),  // Lookup 2: Y=128, c=0
      contrast_adjust(50, 127), // Lookup 3: Y=50, c=127
    ];

    for (x, expected) in expected_values.iter().enumerate() {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, combined_table_index_bits));
      assert_eq!(
        calculated,
        Fr::from(*expected as u64),
        "Failed for lookup {}: expected {}, got {:?}",
        x,
        expected,
        calculated
      );
    }
  }

  // Automatically verifies that evaluate_mle(r) equals evaluate_poly(materialize(), r)
  // This ensures the optimized "tensor product" MLE evaluation in the Verifier matches
  // the naive table materialization in the Prover.
  materialization_mle_parity_test!(
    materialization_parity,
    ContrastSubtableStrategy,
    Fr,
    1 << 16, // M = 2^16 = 65536
    1        // C = 1
  );

  materialization_mle_parity_test!(
    materialization_parity_nonzero_c,
    ContrastSubtableStrategy,
    Fr,
    1 << 16, // M = 2^16 = 65536
    2        // C = 2
  );
}
