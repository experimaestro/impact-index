use impact_index::{
    base::{load_index, ImpactValue, TermImpact, TermIndex},
    builder::Indexer,
    compress::{docid::EliasFanoCompressor, impact::Quantizer, CompressionTransform},
    index::{BlockTermImpactIterator, SparseIndex},
    search::{maxscore::MaxScoreOptions, ScoredDocument, TopScoredDocuments},
    transforms::IndexTransform,
};
use log::{debug, info};
use ntest::assert_about_eq;
use rstest::rstest;

use helpers::documents::{create_document, document_vectors, TestDocument};

use impact_index::base::SearchFn;
use impact_index::search::{maxscore::search_maxscore, wand::search_wand};
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use temp_dir::TempDir;

use std::{collections::HashMap, fmt::Display};

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

struct TestIndex {
    dir: TempDir,
    vocabulary_size: usize,
    all_terms: HashMap<TermIndex, Vec<TermImpact>>,
    indexer: Indexer,
    documents: Vec<TestDocument>,
}

impl TestIndex {
    fn new(
        vocabulary_size: usize,
        document_count: i64,
        lambda_words: f32,
        max_words: usize,
        seed: Option<u64>,
        in_memory_threshold: Option<usize>,
    ) -> Self {
        let dir = TempDir::new().expect("Could not create temporary directory");
        let mut indexer = Indexer::new(&dir.path());

        if let Some(v) = in_memory_threshold {
            indexer.set_in_memory_threshold(v);
        }

        let mut all_terms = HashMap::<TermIndex, Vec<TermImpact>>::new();
        let mut documents = Vec::<TestDocument>::new();
        // let mut rng = thread_rng();
        let mut rng = if let Some(seed) = seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_entropy()
        };

        // Creates documents
        for ix in 0..document_count {
            let doc_id = ix.try_into().unwrap();
            let document = create_document(lambda_words, max_words, vocabulary_size, &mut rng);

            let (terms, values) = document_vectors(&document);

            // Add those to the index
            indexer
                .add(doc_id, &terms, &values)
                .expect("Error while adding terms to the index");

            for term in document.terms.iter() {
                let m = all_terms.get_mut(&term.term_ix);
                let ti = TermImpact {
                    docid: doc_id,
                    value: term.weight,
                };
                match m {
                    Some(p) => {
                        p.push(ti);
                    }
                    None => {
                        let p = vec![ti];
                        all_terms.insert(term.term_ix, p);
                        // eprintln!("Adding {} ({}) in document {}", term.term_ix, ti, ix);
                    }
                }
            }

            documents.push(document);
        }

        // Build the index
        indexer.build().expect("Error while building the index");
        Self {
            dir: dir,
            vocabulary_size: vocabulary_size,
            all_terms: all_terms,
            documents: documents,
            indexer: indexer,
        }
    }
}

#[test]
fn test_index() {
    let mut data = TestIndex::new(100, 1000, 5., 10, None, Some(10));
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

#[rstest]
#[case(true, 100, 1000, 50., 50, 10, None)]
#[case(true, 100, 1000, 50., 50, 1, None)]
// Sparse documents (max 8 postings, 5 in average) )
#[case(true, 500, 500, 5., 8, 10, Some(1))]
fn test_search(
    #[case] in_memory: bool,
    #[case] vocabulary_size: usize,
    #[case] document_count: i64,
    #[case] lambda_words: f32,
    #[case] max_words: usize,
    #[case] top_k: usize,
    #[case] seed: Option<u64>,
    #[values(search_wand, search_maxscore_default)] search_fn: SearchFn,
) {
    use impact_index::builder::load_forward_index;

    init_logger();
    // std::env::set_var("RUST_LOG", "trace");
    debug!("Search test start");
    let mut data = TestIndex::new(
        vocabulary_size,
        document_count,
        lambda_words,
        max_words,
        seed,
        // Use small pages
        Some(10),
    );
    let mut index = if in_memory {
        data.indexer.to_index(true)
    } else {
        load_forward_index(data.dir.path(), true)
    };

    // Builds a query from a document
    let query: HashMap<usize, ImpactValue> = data.documents[10]
        .terms
        .iter()
        .map(|x| (x.term_ix, x.weight as ImpactValue))
        .collect();

    // Search with WAND
    let observed = search_fn(&mut index, &query, top_k);
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

fn check_same_index(
    index_a: &mut dyn BlockTermImpactIterator,
    index_b: &mut dyn BlockTermImpactIterator,
    impact_eps: f64,
) {
    while let Some(a) = index_a.next() {
        let b = index_b.next().expect("Index b contains less entries");
        assert!(a.docid == b.docid);
        assert_about_eq!(a.value, b.value, impact_eps);
    }
}

#[test]
fn test_compressed_index() {
    let mut data = TestIndex::new(100, 1000, 5., 10, None, Some(10));
    let index = data.indexer.to_index(true);

    let dir = TempDir::new().expect("Could not create temporary directory");
    let step = 5. / ((2 << 4) as f64);

    let transform = CompressionTransform {
        max_block_size: 1024,
        doc_ids_compressor: Box::new(EliasFanoCompressor {}),
        impacts_compressor: Box::new(Quantizer::new(4, 0., 5.)),
    };

    transform
        .process(dir.path(), &index)
        .expect("An error occurred");

    let c_index = load_index(dir.path(), true);

    check_same_index(
        c_index.block_iterator(0).as_mut(),
        index.block_iterator(0).as_mut(),
        step,
    );
}
