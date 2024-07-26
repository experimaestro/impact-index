use ndarray::Array;
use rand::{self, RngCore};
use rand_distr::{Distribution, Normal, Poisson};
use std::cmp::min;

use impact_index::base::{ImpactValue, TermIndex};

pub struct TermWeight {
    pub term_ix: TermIndex,
    pub weight: ImpactValue,
}

pub struct TestDocument {
    pub terms: Vec<TermWeight>,
}

pub fn create_document(
    lambda_words: f32,
    max_words: usize,
    vocabulary_size: usize,
    rng: &mut dyn RngCore,
) -> TestDocument {
    let poi = Poisson::new(lambda_words).unwrap();
    let num_words = 1 + poi.sample(rng) as usize;

    let term_ids =
        rand::seq::index::sample(rng, vocabulary_size, min(num_words, max_words)).into_vec();
    let normal = Normal::<f32>::new(1., 1.).unwrap();

    let mut document = TestDocument { terms: Vec::new() };

    for term_ix in term_ids.iter() {
        document.terms.push(TermWeight {
            term_ix: *term_ix,
            weight: (normal.sample(rng).abs() + 1e-5).min(5.),
        })
    }

    return document;
}

pub fn document_vectors(
    document: &TestDocument,
) -> (
    ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>,
    ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>>,
) {
    let terms = Array::from_iter(document.terms.iter().map(|tw| tw.term_ix));
    let values = Array::from_iter(document.terms.iter().map(|tw| tw.weight));

    (terms, values)
}
