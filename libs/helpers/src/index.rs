use log::info;
use ntest::assert_about_eq;
use std::collections::{HashMap, HashSet};

use rand::{rngs::StdRng, SeedableRng};
use temp_dir::TempDir;

use crate::documents::{create_document, document_vectors, TestDocument};
use impact_index::{
    base::{DocId, TermImpact, TermIndex},
    builder::{BuilderOptions, Indexer},
    index::BlockTermImpactIterator,
};

pub struct TestIndex {
    pub dir: TempDir,
    pub vocabulary_size: usize,
    pub all_terms: HashMap<TermIndex, Vec<TermImpact>>,
    pub indexer: Indexer,
    pub documents: Vec<TestDocument>,
}

/// Represents a test index used for creating and managing a temporary index
/// with generated documents and terms.
///
/// # Parameters
/// - `vocabulary_size`: The size of the vocabulary to be used in the index.
/// - `document_count`: The number of documents to be generated and indexed.
/// - `lambda_words`: A parameter controlling the distribution of number of
///   words in the documents.
/// - `max_words`: The maximum number of words.
/// - `seed`: An optional seed for random number generation, ensuring
///   reproducibility.
/// - `options`: Configuration options for the index builder.
/// - `interrupt_indexing`: A vector of document IDs that can be used to
///   interrupt indexing (to test recovery).
///
/// # Returns
/// A new instance of `TestIndex` containing the generated documents, terms, and
/// the built index.
///
/// # Panics
/// - Panics if the temporary directory cannot be created.
/// - Panics if there is an error while adding terms to the index.
/// - Panics if there is an error while building the index.
///
/// # Example
/// ```rust
/// let test_index = TestIndex::new(
///     1000, // vocabulary_size
///     10,   // document_count
///     0.5,  // lambda_words
///     100,  // max_words
///     Some(42), // seed
///     BuilderOptions::default(), // options
///     &vec![] // interrupt_indexing
/// );
/// ```
impl TestIndex {
    pub fn new(
        vocabulary_size: usize,
        document_count: DocId,
        lambda_words: f32,
        max_words: usize,
        seed: Option<u64>,
        options: BuilderOptions,
        interrupt_indexing: &HashSet<DocId>,
    ) -> Self {
        let dir = TempDir::new().expect("Could not create temporary directory");
        let mut indexer = Indexer::new(&dir.path(), &options);

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

            // Add those to the index
            let (terms, values) = document_vectors(&document);
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

            // Interrupts indexing to verify that checkpointing is working
            if interrupt_indexing.contains(&ix) {
                // creates a new indexer
                indexer = Indexer::new(&dir.path(), &options);
                let doc_id = indexer.get_checkpoint_doc_id().unwrap();

                // and add back the documents until the current index
                info!(
                    "Recovering by adding back the document {} -> {}",
                    doc_id + 1,
                    ix
                );
                for ix2 in (doc_id + 1)..=ix {
                    let (terms, values) = document_vectors(&documents[ix2 as usize]);
                    indexer
                        .add(ix2, &terms, &values)
                        .expect("Error while adding terms to the index");
                }
            }
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
