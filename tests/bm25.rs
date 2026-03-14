//! Integration tests for BM25 scoring with BOW index builder.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use impact_index::base::{DocId, ImpactValue, TermIndex};
use impact_index::bow::BOWIndexBuilder;
use impact_index::builder::BuilderOptions;
use impact_index::docmeta::DocMetadata;
use impact_index::index::SparseIndex;
use impact_index::scoring::bm25::BM25Scoring;
use impact_index::scoring::{ScoredIndex, ScoringModel};
use impact_index::search::maxscore::{search_maxscore, MaxScoreOptions};
use impact_index::search::wand::search_wand;
use impact_index::search::{ScoredDocument, TopScoredDocuments};
use impact_index::vocab::analyzer::TextAnalyzer;
use impact_index::vocab::stemmer::{NoStemmer, SnowballStemmer};
use impact_index::vocab::Vocabulary;

fn init_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}

/// Helper to compute BM25 scores from raw data for verification.
fn brute_force_bm25(
    docs: &[(Vec<TermIndex>, Vec<f32>)], // (term_indices, tf_values) per doc
    doc_lengths: &[u32],
    query: &HashMap<TermIndex, f32>,
    k1: f32,
    b: f32,
    top_k: usize,
) -> Vec<ScoredDocument> {
    let num_docs = docs.len();
    let avg_dl: f32 = doc_lengths.iter().map(|&l| l as f32).sum::<f32>() / num_docs as f32;

    // Compute df for each query term
    let mut df: HashMap<TermIndex, u64> = HashMap::new();
    for (terms, _) in docs {
        for &t in terms {
            if query.contains_key(&t) {
                *df.entry(t).or_insert(0) += 1;
            }
        }
    }

    let mut top = TopScoredDocuments::new(top_k);

    for (docid, (terms, values)) in docs.iter().enumerate() {
        let dl = doc_lengths[docid] as f32;
        let mut score: f64 = 0.0;

        for (i, &term_ix) in terms.iter().enumerate() {
            if let Some(&qw) = query.get(&term_ix) {
                let tf = values[i];
                let term_df = *df.get(&term_ix).unwrap_or(&0);

                // IDF
                let n = num_docs as f64;
                let df_f64 = term_df as f64;
                let idf = ((n - df_f64 + 0.5) / (df_f64 + 0.5) + 1.0).ln() as f32;

                // TF normalization
                let tf_norm = (k1 + 1.0) * tf / (k1 * (1.0 - b + b * dl / avg_dl) + tf);

                score += (idf * tf_norm * qw) as f64;
            }
        }

        if score > 0.0 {
            top.add(docid as DocId, score as f32);
        }
    }

    top.into_sorted_vec()
}

#[test]
fn test_bow_builder_and_bm25_search() {
    init_logger();

    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path();

    // Build an index with known TF values
    let mut builder = BOWIndexBuilder::<i32>::new(path, &BuilderOptions::default());

    // Document 0: terms [0, 1, 2] with TF [3, 1, 2]
    builder.add(0, &[0, 1, 2], &[3, 1, 2]).unwrap();
    // Document 1: terms [1, 2, 3] with TF [2, 1, 1]
    builder.add(1, &[1, 2, 3], &[2, 1, 1]).unwrap();
    // Document 2: terms [0, 3, 4] with TF [1, 4, 1]
    builder.add(2, &[0, 3, 4], &[1, 4, 1]).unwrap();
    // Document 3: terms [0, 1] with TF [2, 2]
    builder.add(3, &[0, 1], &[2, 2]).unwrap();
    // Document 4: terms [2, 3, 4] with TF [5, 1, 3]
    builder.add(4, &[2, 3, 4], &[5, 1, 3]).unwrap();

    let (index, doc_meta) = builder.build(true).expect("build failed");

    // Verify doc lengths
    assert_eq!(doc_meta.doc_lengths, vec![6, 4, 6, 4, 9]);

    // Create scored index
    let index: Arc<Box<dyn SparseIndex>> = Arc::new(Box::new(index));
    let doc_meta = Arc::new(doc_meta);
    let scored = ScoredIndex::new(
        index.clone(),
        doc_meta.clone(),
        Box::new(BM25Scoring::new()),
    );

    // Query: terms 0 and 1 with weight 1.0
    let query: HashMap<TermIndex, ImpactValue> = [(0, 1.0), (1, 1.0)].into();

    // Search with WAND
    let wand_results = search_wand(&scored, &query, 5);

    // Search with MaxScore
    let maxscore_results = search_maxscore(&scored, &query, 5, MaxScoreOptions::default());

    // Compute brute-force expected results
    let docs = vec![
        (vec![0, 1, 2], vec![3.0, 1.0, 2.0]),
        (vec![1, 2, 3], vec![2.0, 1.0, 1.0]),
        (vec![0, 3, 4], vec![1.0, 4.0, 1.0]),
        (vec![0, 1], vec![2.0, 2.0]),
        (vec![2, 3, 4], vec![5.0, 1.0, 3.0]),
    ];
    let expected = brute_force_bm25(&docs, &doc_meta.doc_lengths, &query, 1.2, 0.75, 5);

    // Compare
    assert_eq!(
        wand_results.len(),
        expected.len(),
        "WAND result count mismatch"
    );
    for (i, (obs, exp)) in wand_results.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            obs.docid, exp.docid,
            "WAND: doc {} mismatch at position {}",
            obs.docid, i
        );
        assert!(
            (obs.score - exp.score).abs() < 1e-4,
            "WAND: score mismatch at {}: {} vs {}",
            i,
            obs.score,
            exp.score
        );
    }

    assert_eq!(
        maxscore_results.len(),
        expected.len(),
        "MaxScore result count mismatch"
    );
    for (i, (obs, exp)) in maxscore_results.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            obs.docid, exp.docid,
            "MaxScore: doc {} mismatch at position {}",
            obs.docid, i
        );
        assert!(
            (obs.score - exp.score).abs() < 1e-4,
            "MaxScore: score mismatch at {}: {} vs {}",
            i,
            obs.score,
            exp.score
        );
    }
}

#[test]
fn test_bow_builder_with_text_analyzer() {
    init_logger();

    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path();

    let analyzer = TextAnalyzer::new(Box::new(SnowballStemmer::new("english").unwrap()));
    let mut builder =
        BOWIndexBuilder::<i32>::with_analyzer(path, &BuilderOptions::default(), analyzer);

    builder
        .add_text(0, "the quick brown fox jumps over the lazy dog")
        .unwrap();
    builder.add_text(1, "a quick brown cat jumps high").unwrap();
    builder.add_text(2, "the lazy dog sleeps all day").unwrap();

    let query = builder.analyze_query("quick fox");

    let (index, doc_meta) = builder.build(true).expect("build failed");

    // Verify doc lengths
    assert_eq!(doc_meta.doc_lengths[0], 9); // 9 tokens
    assert_eq!(doc_meta.doc_lengths[1], 6); // 6 tokens
    assert_eq!(doc_meta.doc_lengths[2], 6); // 6 tokens

    // Search with BM25
    let index: Arc<Box<dyn SparseIndex>> = Arc::new(Box::new(index));
    let doc_meta = Arc::new(doc_meta);
    let scored = ScoredIndex::new(index, doc_meta, Box::new(BM25Scoring::new()));

    let results = search_wand(&scored, &query, 3);
    assert!(!results.is_empty(), "Should find some results");

    // Doc 0 should score highest (has both "quick" and "fox" stemmed forms)
    assert_eq!(results[0].docid, 0, "Doc 0 should rank first");
}

#[test]
fn test_docmeta_save_load() {
    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path();

    let meta = DocMetadata::from_lengths(vec![10, 20, 5, 15, 30]);
    meta.save(path).unwrap();

    let loaded = DocMetadata::load(path).unwrap();
    assert_eq!(loaded.doc_lengths, vec![10, 20, 5, 15, 30]);
    assert_eq!(loaded.num_docs(), 5);
    assert_eq!(loaded.min_dl(), 5);
    assert!((loaded.avg_dl() - 16.0).abs() < 1e-5);
}

#[test]
fn test_vocabulary_save_load() {
    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path().join("vocab.cbor");

    let mut vocab = Vocabulary::new();
    vocab.get_or_insert("hello");
    vocab.get_or_insert("world");
    vocab.get_or_insert("test");

    vocab.save(&path).unwrap();
    let loaded = Vocabulary::load(&path).unwrap();

    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded.get("hello"), Some(0));
    assert_eq!(loaded.get("world"), Some(1));
    assert_eq!(loaded.get("test"), Some(2));
    assert_eq!(loaded.term(0), "hello");
}

#[test]
fn test_bm25_max_score_safety() {
    // Ensure max_score >= score for all documents
    init_logger();

    let dir = temp_dir::TempDir::new().unwrap();
    let path = dir.path();

    let mut builder = BOWIndexBuilder::<i32>::new(path, &BuilderOptions::default());

    // Create documents with varying lengths
    builder.add(0, &[0, 1], &[1, 1]).unwrap();
    builder.add(1, &[0, 1], &[5, 3]).unwrap();
    builder.add(2, &[0], &[2]).unwrap();
    builder.add(3, &[1, 2], &[1, 10]).unwrap();

    let (index, doc_meta) = builder.build(true).expect("build failed");
    let index: Arc<Box<dyn SparseIndex>> = Arc::new(Box::new(index));

    let mut scoring = BM25Scoring::new();
    let doc_lengths = Arc::new(doc_meta.doc_lengths.clone());
    scoring.initialize(doc_lengths, doc_meta.num_docs());

    // For each term, check max_score >= score for all docs
    for term_ix in 0..index.len() {
        let (_, max_raw) =
            impact_index::index::SparseIndexInformation::value_range(&**index, term_ix);
        let df = index.block_iterator(term_ix).length() as u64;
        let scorer = scoring.term_scorer(df, max_raw);
        let max_score = scorer.max_score(max_raw);

        let mut iter = index.block_iterator(term_ix);
        while iter.next_min_doc_id(0).is_some() {
            let impact = iter.current();
            let score = scorer.score(impact.value, impact.docid);
            assert!(
                max_score >= score - 1e-6,
                "max_score ({}) < score ({}) for term {}, doc {}",
                max_score,
                score,
                term_ix,
                impact.docid
            );
        }
    }
}
