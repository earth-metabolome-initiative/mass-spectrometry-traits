use geometric_traits::prelude::*;
use mass_spectrometry::prelude::*;
use multi_ranged::SimpleRange;

#[test]
fn diagnose_adenine_adenosine() {
    let left: GenericSpectrum =
        GenericSpectrum::adenine().expect("reference spectrum should build");
    let right: GenericSpectrum =
        GenericSpectrum::adenosine().expect("reference spectrum should build");
    let tolerance: f64 = 2.0;
    let mz_power: f64 = 0.0;
    let intensity_power: f64 = 1.0;

    // Compute peak products (same as HungarianCosine)
    let mut left_products: Vec<f64> = Vec::new();
    let mut left_sq_sum: f64 = 0.0;
    for (mz, int) in left.peaks() {
        let w = mz.powf(mz_power) * int.powf(intensity_power);
        left_products.push(w);
        left_sq_sum += w * w;
    }
    let mut right_products: Vec<f64> = Vec::new();
    let mut right_sq_sum: f64 = 0.0;
    for (mz, int) in right.peaks() {
        let w = mz.powf(mz_power) * int.powf(intensity_power);
        right_products.push(w);
        right_sq_sum += w * w;
    }
    let left_norm = left_sq_sum.sqrt();
    let right_norm = right_sq_sum.sqrt();

    println!("left_norm={left_norm:.6e}, right_norm={right_norm:.6e}");

    // Build matching peaks graph
    let graph: RangedCSR2D<u32, u32, SimpleRange<u32>> = left
        .matching_peaks(&right, tolerance)
        .expect("matching graph construction should succeed");

    let max_left: f64 = left_products.iter().cloned().fold(0.0f64, f64::max);
    let max_right: f64 = right_products.iter().cloned().fold(0.0f64, f64::max);

    println!("max_left={max_left:.6e}, max_right={max_right:.6e}");

    // Print all edges with costs
    println!("\n=== Edges and costs ===");
    #[allow(clippy::needless_range_loop)]
    for i in 0..left.len() {
        let row = i as u32;
        for col in graph.sparse_row(row) {
            let j = col as usize;
            let norm_prod = (left_products[i] / max_left) * (right_products[j] / max_right);
            let cost = 1.0f64 + f64::EPSILON - norm_prod;
            let raw_product = left_products[i] * right_products[j];
            println!(
                "  ({i:2}, {col:2}): cost={cost:.15e}, norm_prod={norm_prod:.15e}, raw_product={raw_product:.6e}"
            );
        }
    }

    // Build cost matrix and run LAPMOD
    let map: GenericImplicitValuedMatrix2D<RangedCSR2D<u32, u32, SimpleRange<u32>>, _, f64> =
        GenericImplicitValuedMatrix2D::new(graph.clone(), |(i, j)| {
            1.0f64 + f64::EPSILON
                - (left_products[i as usize] / max_left) * (right_products[j as usize] / max_right)
        });

    let max_edge_cost: f64 = map.max_sparse_value().expect("map is not empty");
    let padding_cost: f64 = 2.1 * max_edge_cost;
    let max_cost: f64 = padding_cost + 1.0;

    println!("\nmax_edge_cost={max_edge_cost:.15e}");
    println!("padding_cost={padding_cost:.15e}");
    println!("max_cost={max_cost:.15e}");

    let matching: Vec<(u32, u32)> = map.jaqaman(padding_cost, max_cost).expect("LAPMOD failed");

    println!("\n=== LAPMOD assignment ===");
    let mut total_product: f64 = 0.0;
    for &(i, j) in &matching {
        let product = left_products[i as usize] * right_products[j as usize];
        total_product += product;
        println!(
            "  left[{i}] (w={:.6e}) -> right[{j}] (w={:.6e}): product={product:.6e}",
            left_products[i as usize], right_products[j as usize]
        );
    }
    let score = total_product / (left_norm * right_norm);
    println!("\ntotal_product={total_product:.6e}");
    println!("score={score:.12}");
    println!("expected score=0.967541239750");

    // What SHOULD the best assignment be? Try row 7 -> col 29
    println!("\n=== Manual check: row 7 -> col 29 ===");
    let p = left_products[7] * right_products[29];
    println!(
        "  left[7]={:.6e} * right[29]={:.6e} = {p:.6e}",
        left_products[7], right_products[29]
    );
    println!(
        "  just this one pair gives score={:.12}",
        p / (left_norm * right_norm)
    );
}
