use impact_index::{
    base::{DocId, ImpactValue, TermIndex},
    builder::BuilderOptions,
    index::SparseIndex,
    search::{maxscore::MaxScoreOptions, ScoredDocument, TopScoredDocuments},
    transforms::IndexTransform,
};
use log::{debug, info};
use rstest::rstest;

use helpers::index::TestIndex;
use impact_index::base::SearchFn;
use impact_index::search::{maxscore::search_maxscore, wand::search_wand};
use rand_distr::{Distribution, LogNormal};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

/// Initialize the logger
fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

trait ApproxEq {
    fn approx_eq(&self, other: &Self, delta: f64) -> bool;
}

impl ApproxEq for ScoredDocument {
    fn approx_eq(&self, other: &Self, delta: f64) -> bool {
        (self.docid == other.docid) && ((self.score - other.score).abs() < (delta as f32))
    }
}

fn vec_compare<T>(observed: &Vec<T>, expected: &Vec<T>)
where
    T: ApproxEq + Display,
{
    assert!(
        observed.len() == expected.len(),
        "Size differ {} vs {}",
        observed.len(),
        expected.len()
    );
    for i in 0..expected.len() {
        assert!(
            observed[i].approx_eq(&expected[i], 1e-2),
            "{}th element differ: {} vs {}",
            i,
            observed[i],
            expected[i]
        );
    }
}

#[test]
fn test_index() {
    let mut data = TestIndex::new(
        100,
        1000,
        5.,
        10,
        None,
        BuilderOptions {
            checkpoint_frequency: 0,
            in_memory_threshold: 10,
        },
        &HashSet::<DocId>::from([]),
    );
    let index = data.indexer.to_index(true);

    eprintln!("Index built in {}", &data.dir.path().display());
    // Verify the index
    for term_ix in 0..data.vocabulary_size {
        let mut iter = match data.all_terms.get(&term_ix) {
            Some(v) => Box::new(v.iter()),
            None => Box::new([].iter()),
        };

        for (ix, observed) in index.block_iterator(term_ix).enumerate() {
            let expected = iter.next().expect(&format!(
                "The index has too many elements for term {}",
                term_ix
            ));
            info!(
                "DocID {} vs {} for term {} [entry {}]",
                expected.docid, observed.docid, term_ix, ix
            );
            assert!(
                expected.docid == observed.docid,
                "Expected doc ID {}, got {} for term {} [entry {}]",
                expected.docid,
                observed.docid,
                term_ix,
                ix
            );
            assert!(expected.value == observed.value);
        }
    }
}

#[test]
fn test_heap() {
    let mut top = TopScoredDocuments::new(3);
    assert!(top.add(0, 0.1) == ImpactValue::NEG_INFINITY);

    let mut max = top.add(1, 0.2);
    assert!(
        max == ImpactValue::NEG_INFINITY,
        "Expected -inf got {}",
        max
    );

    max = top.add(2, 0.3);
    assert!(max == 0.1, "Expected 0.1 got {}", max);

    max = top.add(2, 0.05);
    assert!(max == 0.1, "Expected 0.1 got {}", max);

    max = top.add(3, 0.5);
    assert!(max == 0.2, "Expected 0.2 got {}", max);

    // Further tests
    let top_k = 10;
    let mut rng = rand::thread_rng();
    let log_normal = LogNormal::new(0., 1.).unwrap();

    let mut scored_documents: Vec<ScoredDocument> = Vec::new();
    top = TopScoredDocuments::new(top_k);
    for doc_id in 0..10000 {
        let score = log_normal.sample(&mut rng);
        top.add(doc_id, score);
        scored_documents.push(ScoredDocument {
            docid: doc_id,
            score,
        });
    }

    // Compare
    scored_documents.sort();
    let expected = scored_documents[0..top_k].to_vec();
    let observed = top.into_sorted_vec();

    vec_compare(&observed, &expected);
}

fn search_maxscore_default<'a>(
    index: &'a dyn SparseIndex,
    query: &HashMap<TermIndex, ImpactValue>,
    top_k: usize,
) -> Vec<ScoredDocument> {
    search_maxscore(index, query, top_k, MaxScoreOptions::default())
}

enum IndexType {
    InMemory,
    Disk,
    Split2,
}

#[rstest]
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 0, vec![])]
// Checkpoint at 800, interrupts at 820
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 800, vec![820])]
// Checkpoint every 300 documents, interrupts at 820
#[case(IndexType::InMemory, 100, 1000, 50., 50, 10, None, 300, vec![820])]
#[case(IndexType::InMemory, 100, 1000, 50., 50, 1, None, 0, vec![])]
// Sparse documents (max 8 postings, 5 in average) )
#[case(IndexType::InMemory, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
#[case(IndexType::Disk, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
#[case(IndexType::Split2, 500, 500, 5., 8, 10, Some(1), 0, vec![])]
fn test_search(
    #[case] index_type: IndexType,
    #[case] vocabulary_size: usize,
    #[case] document_count: DocId,
    #[case] lambda_words: f32,
    #[case] max_words: usize,
    #[case] top_k: usize,
    #[case] seed: Option<u64>,
    #[case] checkpoint_frequency: DocId,
    #[case] index_interruptions: Vec<DocId>,
    #[values(search_wand, search_maxscore_default)] search_fn: SearchFn,
) {
    use std::collections::HashSet;

    use impact_index::{
        base::load_index,
        builder::{load_forward_index, BuilderOptions},
        compress::{docid::EliasFanoCompressor, impact::Identity, CompressionTransform},
        transforms::split::SplitIndexTransform,
    };

    init_logger();
    // std::env::set_var("RUST_LOG", "trace");
    debug!("Search test start");
    let index_interruptions_set = HashSet::<DocId>::from_iter(index_interruptions.iter().copied());

    let mut data = TestIndex::new(
        vocabulary_size,
        document_count,
        lambda_words,
        max_words,
        seed,
        // Use small pages
        BuilderOptions {
            in_memory_threshold: 10,
            checkpoint_frequency: checkpoint_frequency,
        },
        &index_interruptions_set,
    );
    let index = match index_type {
        IndexType::InMemory => Box::new(data.indexer.to_index(true)),
        IndexType::Disk => Box::new(load_forward_index(data.dir.path(), true)),
        IndexType::Split2 => {
            let transform = SplitIndexTransform {
                sink: Box::new(CompressionTransform {
                    max_block_size: 1024,
                    doc_ids_compressor_factory: Box::new(EliasFanoCompressor {}),
                    impacts_compressor_factory: Box::new(Identity {}),
                }),
                quantiles: [63. / 64.].to_vec(),
            };
            let split_index_path = data.dir.path().join("split");
            transform
                .process(&split_index_path, &data.indexer.to_index(true))
                .expect("Could not build the new index");
            load_index(&split_index_path, true)
        }
    };

    // Builds a query from a document
    let query: HashMap<usize, ImpactValue> = data.documents[10]
        .terms
        .iter()
        .map(|x| (x.term_ix, x.weight as ImpactValue))
        .collect();

    // Search with WAND
    let observed = search_fn(&*index, &query, top_k);
    eprintln!("(1) observed results");
    for (ix, result) in observed.iter().enumerate() {
        eprintln!(
            " [{}] document {}, score {}",
            ix, result.docid, result.score
        );
    }

    // Searching by iterating
    let mut top = TopScoredDocuments::new(top_k);
    for (doc_id, document) in data.documents.iter().enumerate() {
        let mut score = 0.;
        for tw in document.terms.iter() {
            score += match query.get(&tw.term_ix) {
                Some(s) => *s,
                None => 0.,
            } * (tw.weight as ImpactValue)
        }
        if score > 0. {
            top.add(doc_id.try_into().unwrap(), score);
        }
    }
    eprintln!("(2) expected results");
    let expected = top.into_sorted_vec();
    for (ix, result) in expected.iter().enumerate() {
        eprintln!(
            " [{}] document {}, score {}",
            ix, result.docid, result.score
        );
    }

    vec_compare(&observed, &expected);
}
