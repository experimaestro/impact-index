use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, Criterion};

use helpers::documents::{create_document, document_vectors};
use impact_index::{
    base::SearchFn,
    builder::Indexer,
    search::{
        maxscore::{search_maxscore, MaxScoreOptions},
        wand::search_wand,
    },
};
use log::info;
use rand::thread_rng;
use temp_dir::TempDir;

fn benchmark(c: &mut Criterion, name: &str, search_fn: SearchFn) {
    let mut rng = thread_rng();

    // Create the index
    const FLOPS: f32 = 1.;
    const NUM_DOCS: u64 = 100_000;
    const VOCABULARY_SIZE: usize = 1_000;

    // Compute lambda_words and vocabulary_size
    // p_active = # words / vocabulary_size
    // FLOPS = vocabulary_size x p_active^2 = (# words)^2 / vocabulary_size
    let lambda_words: f32 = f32::sqrt(FLOPS * (VOCABULARY_SIZE as f32));

    info!(
        "Generating an index: FLOPS={}, # docs={}, # tokens={}, # lambda_tokens={}",
        FLOPS, NUM_DOCS, VOCABULARY_SIZE, lambda_words
    );
    // let builder = Indexer::new();

    let dir = TempDir::new().expect("Could not create temporary directory");
    let mut indexer = Indexer::new(&dir.path());

    for doc_id in 0..NUM_DOCS {
        let document = create_document(lambda_words, 100, VOCABULARY_SIZE, &mut rng);
        let (terms, values) = document_vectors(&document);

        // Add those to the index
        indexer
            .add(doc_id, &terms, &values)
            .expect("Error while adding terms to the index");
    }

    indexer.build().expect("Error while building the index");
    let index = indexer.to_index(true);

    let query = HashMap::from([(0, 1.2), (1, 2.3), (2, 3.2), (3, 1.2), (4, 0.7), (5, 2.3)]);

    c.bench_function(name, |b| b.iter(|| search_fn(&index, &query, 1000)));
}

fn benchmark_maxscore(c: &mut Criterion) {
    benchmark(c, "max_score", |index, query, top_k| {
        let options = MaxScoreOptions::default();
        search_maxscore(index, query, top_k, options)
    })
}
fn benchmark_wand(c: &mut Criterion) {
    benchmark(c, "wand", search_wand)
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(500);
    targets = benchmark_maxscore, benchmark_wand
}
criterion_main!(benches);
