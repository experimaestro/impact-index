use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};

use helpers::documents::{create_document, document_vectors};
use rand::thread_rng;
use temp_dir::TempDir;
use xpmir_rust::index::sparse::{builder::Indexer, wand::search_wand};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();

    // Create the index
    const NUM_DOCS: u64 = 10_000;

    // let builder = Indexer::new();

    let dir = TempDir::new().expect("Could not create temporary directory");
    let mut indexer = Indexer::new(&dir.path());

    for doc_id in 0..NUM_DOCS {
        let document = create_document(5., 10, 100, &mut rng);
        let (terms, values) = document_vectors(&document);

        // Add those to the index
        indexer
            .add(doc_id, &terms, &values)
            .expect("Error while adding terms to the index");
    }

    indexer.build().expect("Error while building the index");
    let index = indexer.to_forward_index();

    let query = HashMap::from([(0, 1.2), (5, 2.3)]);

    c.bench_function("wand", |b| b.iter(|| search_wand(&index, &query, 10)));
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(500);
    targets = criterion_benchmark
}
criterion_main!(benches);
