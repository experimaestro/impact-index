
#[cfg(test)]
mod tests {
    use std::{env, collections::{HashMap}, fmt::Display};

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use crate::{index::sparse::{TermImpact, builder::{Indexer, SparseBuilderIndexTrait}, wand::search_wand}, search::{TopScoredDocuments}, base::{TermIndex, ImpactValue}};
    use ndarray::{array, Array};
    use ntest::{timeout, assert_true};
    use rand::thread_rng;
    use rand_distr::{Poisson, Distribution, LogNormal};
    use temp_dir::TempDir;
    use std::cmp::min;

    fn vec_compare<T>(observed: &Vec<T>, expected: &Vec<T>) where T: PartialEq + Display {
        assert!(observed.len() == expected.len(), "Size differ {} vs {}", observed.len(), expected.len());
        for i in 0..expected.len() {
            assert!(observed[i] == expected[i], "{}th element differ: {} vs {}", i, observed[i], expected[i]);
        }
    }

    struct TermWeight {
        term_ix: TermIndex,
        weight: ImpactValue
    }

    struct TestDocument {
        terms: Vec<TermWeight>
    }


    fn createDocument(lambda_words: f32, max_words: usize, vocabulary_size: usize) -> TestDocument {
        let mut rng = thread_rng();
        let poi = Poisson::new(lambda_words).unwrap();
        let num_words = 1 + poi.sample(&mut rng) as usize;

        let term_ids = rand::seq::index::sample(&mut rng, vocabulary_size, min(num_words, max_words)).into_vec();
        let log_normal = LogNormal::new(0., 1.).unwrap();

        let mut document = TestDocument { terms: Vec::new() };

        for term_ix in term_ids.iter() {
            document.terms.push(TermWeight { 
                term_ix: *term_ix, 
                weight: log_normal.sample(&mut rng)
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
        fn new(vocabulary_size: usize, document_count: i64, lambda_words: f32, max_words: usize) -> Self {
            let dir = TempDir::new().expect("Could not create temporary directory");
            let mut indexer = Indexer::new(&dir.path());
    
            let mut all_terms = HashMap::<TermIndex, Vec::<TermImpact>>::new();
            let mut documents = Vec::<TestDocument>::new();
    
            // Creates documents
            for ix in 0..document_count { 
                let doc_id = ix.try_into().unwrap();
                let document = createDocument(lambda_words, max_words, vocabulary_size);
    
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
        let mut data = TestIndex::new(100, 1000, 5., 10);
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
    fn test_search() {
        let mut data = TestIndex::new(200, 10000, 10., 50);
        let mut index = data.indexer.to_forward_index();

        // let index = data.indexer.to_forward_index();
        let query = HashMap::<usize, f64>::from([
            (1, 0.4),
            (2, 0.2)
        ]);
        let observed = search_wand(&mut index, &query, 10);
        eprintln!("Results are");
        for result in observed.iter() {
            eprintln!("document {}, score {}", result.docid, result.score);
        }

        // Searching by iterating
        let mut top = TopScoredDocuments::new(10);
        for (doc_id, document) in data.documents.iter().enumerate() {
            let mut score = 0.;
            for tw in document.terms.iter() {
                score += match query.get(&tw.term_ix) {
                    Some(s) => *s,
                    None => 0.
                } * (tw.weight as f64)
            }
            top.add(doc_id.try_into().unwrap(), score);
        }
        eprintln!("Results are");
        let expected = top.into_sorted_vec();
        for result in expected.iter() {
            eprintln!("document {}, score {}", result.docid, result.score);
        }

        vec_compare(&expected, &observed);

    }
}
