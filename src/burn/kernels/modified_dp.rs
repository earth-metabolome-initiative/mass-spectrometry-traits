//! `#[cube]` helpers shared by every modified (precursor-shifted) similarity.
//!
//! The modified variants build a small conflict graph over candidate peak-pair
//! matches: a node is a peak, an edge is a candidate match `(left_peak,
//! right_peak)`, and two edges conflict when they share either endpoint
//! (a peak cannot be assigned to two partners). Each node touches at most
//! two edges, so the conflict graph decomposes into paths. The maximum-weight
//! independent set on each path is found by a small DP and used to pick the
//! optimal subset of matches.
//!
//! The pieces that are entirely metric-agnostic live here: neighbour-list
//! construction and traversal, the candidate-collection sweeps (direct +
//! precursor-shifted), an insertion-sort dedupe, and the conflict-graph
//! population. The per-pair benefit + score accumulation stays in each
//! metric scorer because the formula differs (cosine vs entropy).

use burn_cubecl::cubecl::prelude::*;

/// Record that `neighbor` is adjacent to `edge` in the conflict graph.
///
/// Each node carries up to two neighbour slots (`neighbor_a`, `neighbor_b`),
/// matching the property that no peak appears in more than two candidate
/// edges (the conflict-graph node degree). Idempotent, calling twice with
/// the same `(edge, neighbor)` is a no-op.
#[cube]
pub fn insert_modified_neighbor(
    neighbor_a: &mut Array<u32>,
    neighbor_b: &mut Array<u32>,
    edge: u32,
    neighbor: u32,
    invalid: u32,
) {
    let edge_index = edge as usize;
    if neighbor_a[edge_index] == invalid {
        neighbor_a[edge_index] = neighbor;
    } else if neighbor_a[edge_index] != neighbor {
        neighbor_b[edge_index] = neighbor;
    }
}

/// Return the first neighbour slot that is set and differs from `from`,
/// or `invalid` when no such neighbour exists.
///
/// Used to walk a path component without backtracking: `from` is the previous
/// node, the returned value is the next node, and the chain ends when both
/// slots either contain `invalid` or contain `from`.
#[cube]
pub fn first_modified_neighbor_not_from(
    neighbor_a: u32,
    neighbor_b: u32,
    from: u32,
    invalid: u32,
) -> u32 {
    let mut next = invalid;
    if neighbor_a != invalid && neighbor_a != from {
        next = neighbor_a;
    } else if neighbor_b != invalid && neighbor_b != from {
        next = neighbor_b;
    }
    next
}

/// Run the direct and (if precursors differ by more than `tolerance`)
/// precursor-shifted two-pointer sweeps, appending `(left_peak, right_peak)`
/// candidate matches into `candidate_left` / `candidate_right`.
///
/// Returns the number of candidates written. The `products` arrays are
/// consulted only to skip padding peaks (`product <= 0`), the actual scoring
/// happens in the metric scorer.
#[cube]
pub fn collect_modified_candidates<F: Float>(
    left_mz: &Tensor<F>,
    left_products: &Array<F>,
    left_row: usize,
    left_peaks: usize,
    left_precursor_value: F,
    right_mz: &Tensor<F>,
    right_products: &Array<F>,
    right_row: usize,
    right_peaks: usize,
    right_precursor_value: F,
    tolerance: F,
    candidate_left: &mut Array<u32>,
    candidate_right: &mut Array<u32>,
    #[comptime] candidate_capacity: u32,
) -> u32 {
    let zero = F::new(0.0_f32);
    let candidate_capacity_usize = comptime!(candidate_capacity as usize);
    let mut candidate_count = 0u32;

    let mut right_cursor = 0usize;
    for left_peak in 0..left_peaks {
        if left_products[left_peak] > zero {
            let left_value = left_mz[left_row * left_mz.stride(0) + left_peak * left_mz.stride(1)];
            while right_cursor < right_peaks && right_products[right_cursor] <= zero {
                right_cursor += 1;
            }
            while right_cursor < right_peaks {
                if right_products[right_cursor] <= zero {
                    right_cursor += 1;
                } else {
                    let right_value = right_mz
                        [right_row * right_mz.stride(0) + right_cursor * right_mz.stride(1)];
                    let delta = left_value - right_value;
                    if delta > tolerance {
                        right_cursor += 1;
                    } else if delta.abs() <= tolerance {
                        if (candidate_count as usize) < candidate_capacity_usize {
                            candidate_left[candidate_count as usize] = left_peak as u32;
                            candidate_right[candidate_count as usize] = right_cursor as u32;
                            candidate_count += 1u32;
                        }
                        right_cursor += 1;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    if right_precursor_value < left_precursor_value - tolerance
        || right_precursor_value > left_precursor_value + tolerance
    {
        let mut shifted_right_cursor = 0usize;
        for left_peak in 0..left_peaks {
            if left_products[left_peak] > zero {
                let left_value = left_mz
                    [left_row * left_mz.stride(0) + left_peak * left_mz.stride(1)]
                    - left_precursor_value;
                while shifted_right_cursor < right_peaks
                    && right_products[shifted_right_cursor] <= zero
                {
                    shifted_right_cursor += 1;
                }
                while shifted_right_cursor < right_peaks {
                    if right_products[shifted_right_cursor] <= zero {
                        shifted_right_cursor += 1;
                    } else {
                        let right_value = right_mz[right_row * right_mz.stride(0)
                            + shifted_right_cursor * right_mz.stride(1)]
                            - right_precursor_value;
                        let delta = left_value - right_value;
                        if delta > tolerance {
                            shifted_right_cursor += 1;
                        } else if delta.abs() <= tolerance {
                            if (candidate_count as usize) < candidate_capacity_usize {
                                candidate_left[candidate_count as usize] = left_peak as u32;
                                candidate_right[candidate_count as usize] =
                                    shifted_right_cursor as u32;
                                candidate_count += 1u32;
                            }
                            shifted_right_cursor += 1;
                        } else {
                            break;
                        }
                    }
                }
            }
        }
    }

    candidate_count
}

/// In-place insertion-sort + dedupe of the candidate edge list.
///
/// Sort order: by `(candidate_left, candidate_right)` ascending. Duplicates
/// are dropped (only the first occurrence is kept). Returns the new count.
#[cube]
pub fn sort_and_dedupe_modified_candidates(
    candidate_left: &mut Array<u32>,
    candidate_right: &mut Array<u32>,
    candidate_count: u32,
) -> u32 {
    let count = candidate_count as usize;
    let mut unique_count = 0u32;
    if count > 0usize {
        for sort_index in 1..count {
            let key_left = candidate_left[sort_index];
            let key_right = candidate_right[sort_index];
            let mut insert_index = sort_index;
            while insert_index > 0usize {
                let previous = insert_index - 1usize;
                let previous_left = candidate_left[previous];
                let previous_right = candidate_right[previous];
                if previous_left > key_left
                    || (previous_left == key_left && previous_right > key_right)
                {
                    candidate_left[insert_index] = previous_left;
                    candidate_right[insert_index] = previous_right;
                    insert_index -= 1;
                } else {
                    break;
                }
            }
            candidate_left[insert_index] = key_left;
            candidate_right[insert_index] = key_right;
        }

        for read_index in 0..count {
            let current_left = candidate_left[read_index];
            let current_right = candidate_right[read_index];
            if read_index == 0usize
                || current_left != candidate_left[read_index - 1usize]
                || current_right != candidate_right[read_index - 1usize]
            {
                candidate_left[unique_count as usize] = current_left;
                candidate_right[unique_count as usize] = current_right;
                unique_count += 1u32;
            }
        }
    }
    unique_count
}

/// Build the per-peak slot lists and per-edge neighbour arrays for the
/// conflict graph, given the deduplicated candidate edge list.
///
/// `left_slot_a/b`, `right_slot_a/b` (size `max_peaks`), `neighbor_a/b`,
/// `visited` (size `candidate_capacity`) must be allocated by the caller and
/// will be fully initialised by this function. `visited` is reset to zero.
#[cube]
pub fn build_modified_conflict_graph(
    candidate_left: &Array<u32>,
    candidate_right: &Array<u32>,
    candidate_count: u32,
    left_slot_a: &mut Array<u32>,
    left_slot_b: &mut Array<u32>,
    right_slot_a: &mut Array<u32>,
    right_slot_b: &mut Array<u32>,
    neighbor_a: &mut Array<u32>,
    neighbor_b: &mut Array<u32>,
    visited: &mut Array<u32>,
    invalid: u32,
    #[comptime] max_peaks: u32,
) {
    let max_peaks_usize = comptime!(max_peaks as usize);
    let count = candidate_count as usize;

    for peak in 0..max_peaks_usize {
        left_slot_a[peak] = invalid;
        left_slot_b[peak] = invalid;
        right_slot_a[peak] = invalid;
        right_slot_b[peak] = invalid;
    }

    for edge in 0..count {
        neighbor_a[edge] = invalid;
        neighbor_b[edge] = invalid;
        visited[edge] = 0u32;

        let left_peak = candidate_left[edge] as usize;
        if left_slot_a[left_peak] == invalid {
            left_slot_a[left_peak] = edge as u32;
        } else {
            left_slot_b[left_peak] = edge as u32;
        }

        let right_peak = candidate_right[edge] as usize;
        if right_slot_a[right_peak] == invalid {
            right_slot_a[right_peak] = edge as u32;
        } else {
            right_slot_b[right_peak] = edge as u32;
        }
    }

    for peak in 0..max_peaks_usize {
        let left_a = left_slot_a[peak];
        let left_b = left_slot_b[peak];
        if left_a != invalid && left_b != invalid {
            insert_modified_neighbor(neighbor_a, neighbor_b, left_a, left_b, invalid);
            insert_modified_neighbor(neighbor_a, neighbor_b, left_b, left_a, invalid);
        }

        let right_a = right_slot_a[peak];
        let right_b = right_slot_b[peak];
        if right_a != invalid && right_b != invalid {
            insert_modified_neighbor(neighbor_a, neighbor_b, right_a, right_b, invalid);
            insert_modified_neighbor(neighbor_a, neighbor_b, right_b, right_a, invalid);
        }
    }
}
