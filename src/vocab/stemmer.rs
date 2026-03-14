//! Pluggable stemmer trait and implementations.
//!
//! Provides [`Stemmer`] trait for word stemming, with [`SnowballStemmer`]
//! (wrapping `rust-stemmers`) and [`NoStemmer`] (identity) implementations.

use rust_stemmers::{Algorithm, Stemmer as RustStemmer};

/// Trait for stemmers that reduce words to their root form.
pub trait Stemmer: Send + Sync {
    /// Stem a word to its root form.
    fn stem(&self, word: &str) -> String;
}

/// Snowball stemmer wrapping the `rust-stemmers` crate.
pub struct SnowballStemmer {
    stemmer: RustStemmer,
}

impl SnowballStemmer {
    /// Create a new Snowball stemmer for the given language.
    ///
    /// Supported languages: "arabic", "danish", "dutch", "english", "finnish",
    /// "french", "german", "greek", "hungarian", "italian", "norwegian",
    /// "portuguese", "romanian", "russian", "spanish", "swedish", "tamil",
    /// "turkish".
    pub fn new(language: &str) -> Result<Self, String> {
        let algorithm = match language.to_lowercase().as_str() {
            "arabic" => Algorithm::Arabic,
            "danish" => Algorithm::Danish,
            "dutch" => Algorithm::Dutch,
            "english" => Algorithm::English,
            "finnish" => Algorithm::Finnish,
            "french" => Algorithm::French,
            "german" => Algorithm::German,
            "greek" => Algorithm::Greek,
            "hungarian" => Algorithm::Hungarian,
            "italian" => Algorithm::Italian,
            "norwegian" => Algorithm::Norwegian,
            "portuguese" => Algorithm::Portuguese,
            "romanian" => Algorithm::Romanian,
            "russian" => Algorithm::Russian,
            "spanish" => Algorithm::Spanish,
            "swedish" => Algorithm::Swedish,
            "tamil" => Algorithm::Tamil,
            "turkish" => Algorithm::Turkish,
            other => return Err(format!("Unknown stemmer language: {}", other)),
        };
        Ok(Self {
            stemmer: RustStemmer::create(algorithm),
        })
    }
}

impl Stemmer for SnowballStemmer {
    fn stem(&self, word: &str) -> String {
        self.stemmer.stem(word).to_string()
    }
}

/// Identity stemmer (no-op).
pub struct NoStemmer;

impl Stemmer for NoStemmer {
    fn stem(&self, word: &str) -> String {
        word.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snowball_english() {
        let stemmer = SnowballStemmer::new("english").unwrap();
        assert_eq!(stemmer.stem("running"), "run");
        assert_eq!(stemmer.stem("jumps"), "jump");
        assert_eq!(stemmer.stem("easily"), "easili");
    }

    #[test]
    fn test_no_stemmer() {
        let stemmer = NoStemmer;
        assert_eq!(stemmer.stem("running"), "running");
    }
}
