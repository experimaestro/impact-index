//! Text analysis pipeline: tokenize, lowercase, stem, vocabulary lookup.
//!
//! [`TextAnalyzer`] provides document and query analysis:
//! - Document analysis grows the vocabulary as new terms are encountered
//! - Query analysis does NOT grow the vocabulary (unknown terms are skipped)

use std::collections::HashMap;
use std::path::Path;

use crate::base::TermIndex;

use super::stemmer::Stemmer;
use super::Vocabulary;

/// Full text analysis pipeline: tokenize -> lowercase -> stem -> vocabulary lookup.
pub struct TextAnalyzer {
    vocab: Vocabulary,
    stemmer: Box<dyn Stemmer>,
}

impl TextAnalyzer {
    /// Create a new analyzer with the given stemmer.
    pub fn new(stemmer: Box<dyn Stemmer>) -> Self {
        Self {
            vocab: Vocabulary::new(),
            stemmer,
        }
    }

    /// Create from an existing vocabulary and stemmer.
    pub fn with_vocab(vocab: Vocabulary, stemmer: Box<dyn Stemmer>) -> Self {
        Self { vocab, stemmer }
    }

    /// Tokenize text into words using Unicode word boundaries.
    ///
    /// Splits on whitespace and punctuation, lowercases, and filters out
    /// tokens that are purely punctuation or whitespace.
    fn tokenize<'a>(&self, text: &'a str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Analyze document text: tokenize, stem, compute TF, grow vocabulary.
    ///
    /// Returns `(term_indices, tf_values)` suitable for indexing.
    pub fn analyze_doc(&mut self, text: &str) -> (Vec<TermIndex>, Vec<f32>) {
        let tokens = self.tokenize(text);

        // Count term frequencies
        let mut tf_map: HashMap<String, f32> = HashMap::new();
        for token in &tokens {
            let stemmed = self.stemmer.stem(token);
            *tf_map.entry(stemmed).or_insert(0.0) += 1.0;
        }

        // Convert to term indices (growing vocabulary)
        let mut term_indices = Vec::with_capacity(tf_map.len());
        let mut tf_values = Vec::with_capacity(tf_map.len());
        for (term, tf) in tf_map {
            let idx = self.vocab.get_or_insert(&term);
            term_indices.push(idx);
            tf_values.push(tf);
        }

        (term_indices, tf_values)
    }

    /// Analyze query text: tokenize, stem, lookup in vocabulary.
    ///
    /// Does NOT grow vocabulary — unknown terms are skipped.
    /// Returns a map from TermIndex to TF (for boosting).
    pub fn analyze_query(&self, text: &str) -> HashMap<TermIndex, f32> {
        let tokens = self.tokenize(text);
        let mut query: HashMap<TermIndex, f32> = HashMap::new();

        for token in &tokens {
            let stemmed = self.stemmer.stem(token);
            if let Some(idx) = self.vocab.get(&stemmed) {
                *query.entry(idx).or_insert(0.0) += 1.0;
            }
        }

        query
    }

    /// Get a reference to the vocabulary.
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }

    /// Save the vocabulary to `vocab.cbor` in the given directory.
    pub fn save_vocab(&self, dir: &Path) -> std::io::Result<()> {
        self.vocab.save(&dir.join("vocab.cbor"))
    }

    /// Load vocabulary from `vocab.cbor` in the given directory, with a stemmer.
    pub fn load(dir: &Path, stemmer: Box<dyn Stemmer>) -> std::io::Result<Self> {
        let vocab = Vocabulary::load(&dir.join("vocab.cbor"))?;
        Ok(Self { vocab, stemmer })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::stemmer::NoStemmer;

    #[test]
    fn test_analyze_doc() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let (terms, values) = analyzer.analyze_doc("the quick brown fox the quick");

        // Should have 4 unique terms
        assert_eq!(terms.len(), 4);
        assert_eq!(values.len(), 4);

        // "the" and "quick" should have tf=2
        let the_idx = analyzer.vocab().get("the").unwrap();
        let quick_idx = analyzer.vocab().get("quick").unwrap();
        let pos_the = terms.iter().position(|&t| t == the_idx).unwrap();
        let pos_quick = terms.iter().position(|&t| t == quick_idx).unwrap();
        assert_eq!(values[pos_the], 2.0);
        assert_eq!(values[pos_quick], 2.0);
    }

    #[test]
    fn test_analyze_query_no_growth() {
        let mut analyzer = TextAnalyzer::new(Box::new(NoStemmer));
        let _ = analyzer.analyze_doc("hello world");
        let vocab_size_before = analyzer.vocab().len();

        let query = analyzer.analyze_query("hello unknown");
        // Vocabulary should not grow
        assert_eq!(analyzer.vocab().len(), vocab_size_before);
        // Only "hello" should be in the query
        assert_eq!(query.len(), 1);
        let hello_idx = analyzer.vocab().get("hello").unwrap();
        assert!(query.contains_key(&hello_idx));
    }
}
