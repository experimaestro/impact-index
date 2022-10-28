

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use crate::{index::sparse::{TermImpact, builder::{Indexer, SparseBuilderIndexTrait, load_forward_index}, wand::search_wand}, search::{TopScoredDocuments, ScoredDocument}, base::{TermIndex, ImpactValue}};

    use ndarray::{array, Array};
    use ntest::{timeout, assert_true};
    use rand::{thread_rng, rngs::{ThreadRng, StdRng}, SeedableRng, RngCore};
    use rand_distr::{Poisson, Distribution, LogNormal};
    use temp_dir::TempDir;
    use std::cmp::min;

    use std::{env, collections::{HashMap}, fmt::Display};
    use std::iter::FromIterator;

    trait ApproxEq {
        fn approx_eq(&self, other: &Self, delta: f64) -> bool;
    }

    impl Clone for ScoredDocument {
        fn clone(&self) -> Self {
            Self { docid: self.docid.clone(), score: self.score.clone() }
        }
    }

    impl ApproxEq for ScoredDocument {
        fn approx_eq(&self, other: &Self, delta: f64) -> bool {
            (self.docid == other.docid) && ((self.score - other.score).abs() < delta)
        }
    }

    fn vec_compare<T>(observed: &Vec<T>, expected: &Vec<T>) where T: ApproxEq + Display {
        assert!(observed.len() == expected.len(), "Size differ {} vs {}", observed.len(), expected.len());
        for i in 0..expected.len() {
            assert!(observed[i].approx_eq(&expected[i], 1e-5), "{}th element differ: {} vs {}", i, observed[i], expected[i]);
        }
    }

    struct TermWeight {
        term_ix: TermIndex,
        weight: ImpactValue
    }

    struct TestDocument {
        terms: Vec<TermWeight>
    }


    fn createDocument(lambda_words: f32, max_words: usize, vocabulary_size: usize, rng: &mut dyn RngCore) -> TestDocument {
        let poi = Poisson::new(lambda_words).unwrap();
        let num_words = 1 + poi.sample(rng) as usize;

        let term_ids = rand::seq::index::sample(rng, vocabulary_size, min(num_words, max_words)).into_vec();
        let log_normal = LogNormal::new(0., 1.).unwrap();

        let mut document = TestDocument { terms: Vec::new() };

        for term_ix in term_ids.iter() {
            document.terms.push(TermWeight { 
                term_ix: *term_ix, 
                weight: log_normal.sample(rng)
            })
        }

        return document
    }

    struct TestIndex {
        dir: TempDir,
        vocabulary_size: usize,
        all_terms: HashMap::<TermIndex, Vec::<TermImpact>>,
        indexer: Indexer,
        documents: Vec::<TestDocument>
    }

    impl TestIndex {
        fn new(vocabulary_size: usize, document_count: i64, lambda_words: f32, max_words: usize, seed: Option<u64>) -> Self {
            let dir = TempDir::new().expect("Could not create temporary directory");
            let mut indexer = Indexer::new(&dir.path());
    
            let mut all_terms = HashMap::<TermIndex, Vec::<TermImpact>>::new();
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
                let document = createDocument(lambda_words, max_words, vocabulary_size, &mut rng);
    
                let terms = Array::from_iter(document.terms.iter().map(
                    |tw| tw.term_ix
                ));
                let values = Array::from_iter(document.terms.iter().map(
                    |tw| tw.weight
                ));
    
                // Add those to the index
                indexer.add(doc_id, &terms, &values).expect("Error while adding terms to the index");
    
                for term in document.terms.iter() {
                    let m = all_terms.get_mut(&term.term_ix);
                    let ti = TermImpact { docid: doc_id, value: term.weight};
                    match m {
                        Some(p) => {
                            p.push(ti);
                        },
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
                indexer: indexer
            }
        }
    }
    
    #[test]
    fn test_index() {
        let mut data = TestIndex::new(100, 1000, 5., 10, None);
        let index = data.indexer.to_forward_index();

        eprintln!("Index built in {}", &data.dir.path().display());
        // Verify the index
        for term_ix in 0..data.vocabulary_size {

            // eprintln!("Looking at term with index {}", term_ix);
            let mut iter = match data.all_terms.get(&term_ix) {
                Some(v) => Box::new(v.iter()),
                None => Box::new([].iter())
            };


            for observed in index.iter(term_ix) {
                let expected = iter.next().expect(&format!("The index has too many elements for term {}", term_ix));
                assert!(expected.docid == observed.docid);
                assert!(expected.value == observed.value);
            }

        }
    }

    #[test]
    fn test_heap() {
        let mut top = TopScoredDocuments::new(3);
        assert!(top.add(0, 0.1) == f64::NEG_INFINITY);

        let mut max = top.add(1, 0.2);
        assert!(max == f64::NEG_INFINITY, "Expected -inf got {}", max);

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

        let mut scored_documents : Vec::<ScoredDocument> = Vec::new();
        top = TopScoredDocuments::new(top_k);
        for doc_id in 0..10000 {
            let score = log_normal.sample(&mut rng);
            top.add(doc_id, score);
            scored_documents.push(ScoredDocument { docid: doc_id, score });
        }


        // Compare
        scored_documents.sort();
        let expected = scored_documents[0..top_k].to_vec();
        let observed = top.into_sorted_vec();

        vec_compare(&observed, &expected);

    }

    #[rstest]
    #[case(true, 100, 1000, 50., 50, 10, None)]
    #[case(true, 100, 1000, 50., 50, 1, None)]
    // Sparse documents (max 8 postings, 5 in average) )
    #[case(true, 500, 500, 5., 8, 10, Some(1))]
    fn test_search(#[case] in_memory: bool,
        #[case] vocabulary_size: usize, 
        #[case] document_count: i64, 
        #[case] lambda_words: f32,
        #[case] max_words: usize,
        #[case] top_k: usize,
        #[case] seed: Option<u64>
    ) {
        let mut data = TestIndex::new(vocabulary_size, document_count, lambda_words, max_words, seed);
        let mut index = if in_memory {
            data.indexer.to_forward_index()
        } else {
            load_forward_index(data.dir.path())
        };
            

        // Builds a query from a document
        let query: HashMap::<usize, f64> = data.documents[10].terms.iter().map(|x| (x.term_ix, x.weight as f64)).collect();

        // Search with WAND
        let observed = search_wand(&mut index, &query, top_k);
        eprintln!("(1) observed results");
        for (ix, result) in observed.iter().enumerate() {
            eprintln!(" [{}] document {}, score {}", ix, result.docid, result.score);
        }

        // Searching by iterating
        let mut top = TopScoredDocuments::new(top_k);
        for (doc_id, document) in data.documents.iter().enumerate() {
            let mut score = 0.;
            for tw in document.terms.iter() {
                score += match query.get(&tw.term_ix) {
                    Some(s) => *s,
                    None => 0.
                } * (tw.weight as f64)
            }
            if score > 0. {
                top.add(doc_id.try_into().unwrap(), score);
            }
        }
        eprintln!("(2) expected results");
        let expected = top.into_sorted_vec();
        for (ix, result) in expected.iter().enumerate() {
            eprintln!(" [{}] document {}, score {}", ix, result.docid, result.score);
        }

        vec_compare(&observed, &expected);

    }
}
