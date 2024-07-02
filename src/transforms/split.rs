//! Splits an index for a term

use std::fs::create_dir;
use std::path::Path;
use std::sync::Mutex;

use crate::base::{load_index, save_index, ImpactValue, IndexLoader, Len, TermImpact, TermIndex};
use crate::index::SparseIndexView;
use crate::{
    index::{BlockTermImpactIterator, SparseIndex},
    transforms::IndexTransform,
};

use serde::{Deserialize, Serialize};

pub(crate) struct SplitIndexTransform {
    pub sink: Box<dyn IndexTransform>,
    pub quantiles: Vec<f64>,
}

impl IndexTransform for SplitIndexTransform {
    fn process(
        &self,
        path: &std::path::Path,
        index: &dyn SparseIndexView,
    ) -> Result<(), std::io::Error> {
        if !path.is_dir() {
            create_dir(path)?;
        }
        let inner_path = path.join("inner");

        let split_view = SplitIndexView::new(index, &self.quantiles);
        self.sink.process(inner_path.as_path(), &split_view)?;

        let index = SplitIndexLoader {
            splits: self.quantiles.len() + 1,
        };

        save_index(Box::new(index), path)
    }
}

struct SplitIndex {
    /// Inner index that contains the postings
    inner: Box<dyn SparseIndex>,

    /// Number of split per term
    splits: usize,
}

#[derive(Serialize, Deserialize)]
struct SplitIndexLoader {
    splits: usize,
}

#[typetag::serde]
impl IndexLoader for SplitIndexLoader {
    fn into_index(self: Box<Self>, path: &Path, in_memory: bool) -> Box<dyn SparseIndex> {
        let inner = load_index(&path.join("inner"), in_memory);
        Box::new(SplitIndex {
            inner,
            splits: self.splits,
        })
    }
}

struct SplitIndexTermIterator<'a> {
    iterators: Vec<Box<dyn BlockTermImpactIterator + 'a>>,
}

impl<'a> BlockTermImpactIterator for SplitIndexTermIterator<'a> {
    fn next_min_doc_id(&mut self, doc_id: crate::base::DocId) -> bool {
        todo!()
    }

    fn current(&self) -> TermImpact {
        todo!()
    }

    fn max_value(&self) -> ImpactValue {
        todo!()
    }

    fn max_doc_id(&self) -> crate::base::DocId {
        todo!()
    }

    fn length(&self) -> usize {
        todo!()
    }
}

impl SparseIndex for SplitIndex {
    fn block_iterator(
        &self,
        term_ix: crate::base::TermIndex,
    ) -> Box<dyn BlockTermImpactIterator + '_> {
        todo!(); //Box::new(SplitPostingIterator::new(self, term_ix))
    }

    fn block_iterators(&self, term_ix: TermIndex) -> Vec<Box<dyn BlockTermImpactIterator + '_>> {
        let mut v: Vec<Box<dyn BlockTermImpactIterator>> = Vec::new();

        for i in 1..self.splits {
            let mut iterators = Vec::new();
            for j in 1..i {
                iterators.push(self.inner.block_iterator(term_ix * self.splits + j - 1));
            }
            v.push(Box::new(SplitIndexTermIterator {
                iterators: iterators,
            }));
        }
        v
    }
}

impl Len for SplitIndex {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// View on the index
struct SplitIndexView<'a> {
    /// Inner index that contains the postings
    source: &'a dyn SparseIndexView,

    /// Split quantiles (just one value if using two posting lists per term)
    quantiles: &'a Vec<f64>,

    /// The (cached) split values
    thresholds: Mutex<Vec<Vec<ImpactValue>>>,
}

impl<'a> SplitIndexView<'a> {
    pub fn new(source: &'a dyn SparseIndexView, quantiles: &'a Vec<f64>) -> Self {
        let mut thresholds = Vec::new();
        for _ in 1..source.len() {
            thresholds.push(Vec::new());
        }

        Self {
            source: source,
            quantiles: quantiles,
            thresholds: Mutex::new(thresholds),
        }
    }
}

struct SplitIndexViewIterator<'a> {
    iterator: Box<dyn Iterator<Item = TermImpact> + 'a>,
    min: ImpactValue,
    max: ImpactValue,
}

impl<'a> Iterator for SplitIndexViewIterator<'a> {
    type Item = TermImpact;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(posting) = self.iterator.next() {
            if (posting.value >= self.min) && (posting.value < self.max) {
                return Some(TermImpact {
                    docid: posting.docid,
                    value: posting.value,
                });
            }
        }
        None
    }
}

impl<'a> SparseIndexView for SplitIndexView<'a> {
    fn iterator<'b>(&'b self, term_ix: TermIndex) -> Box<dyn Iterator<Item = TermImpact> + 'b> {
        // Source term and quantile indices
        let source_term_ix = term_ix / (self.quantiles.len() + 1);
        let quantile_ix = term_ix % (self.quantiles.len() + 1);

        let thresholds = &mut self.thresholds.lock().unwrap();
        let term_thresholds = &mut thresholds[source_term_ix];

        // Computes the term threshold if not in cache
        if term_thresholds.len() == 0 {
            let mut values: Vec<ImpactValue> = self
                .source
                .iterator(term_ix)
                .map(|posting| posting.value)
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            term_thresholds.push(0.);
            for q in self.quantiles {
                let ix = (q * values.len() as f64).trunc() as usize;
                let mut threshold = 0.;
                if ix < values.len() {
                    threshold = values[ix];
                }
                term_thresholds.push(threshold);
            }
            term_thresholds.push(ImpactValue::INFINITY);
        }

        // Returns the iterator
        Box::new(SplitIndexViewIterator {
            iterator: self.source.iterator(source_term_ix),
            min: term_thresholds[quantile_ix],
            max: term_thresholds[quantile_ix + 1],
        })
    }
}

impl<'a> Len for SplitIndexView<'a> {
    fn len(&self) -> usize {
        self.source.len() * (self.quantiles.len() + 1)
    }
}
