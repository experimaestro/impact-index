use ntest::assert_about_eq;
use std::collections::HashMap;

use rand::{rngs::StdRng, SeedableRng};
use temp_dir::TempDir;

use crate::documents::{create_document, document_vectors, TestDocument};
use impact_index::{
    base::{TermImpact, TermIndex},
    builder::Indexer,
    index::BlockTermImpactIterator,
};

pub struct TestIndex {
    pub dir: TempDir,
    pub vocabulary_size: usize,
    pub all_terms: HashMap<TermIndex, Vec<TermImpact>>,
    pub indexer: Indexer,
    pub documents: Vec<TestDocument>,
}

impl TestIndex {
    pub fn new(
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

/// Test if the index is the same
pub fn check_same_index(
    expected: &mut dyn BlockTermImpactIterator,
    observed: &mut dyn BlockTermImpactIterator,
    impact_eps: f64,
) {
    while let Some(a) = expected.next() {
        let b = observed.next().expect("Index b contains less entries");
        assert!(
            a.docid == b.docid,
            "Expected doc ID {}, got {}",
            a.docid,
            b.docid
        );
        if impact_eps > 0. {
            assert_about_eq!(a.value, b.value, impact_eps);
        } else {
            assert_eq!(a.value, b.value);
        }
    }
}
