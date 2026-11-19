//! BERT tokenizer — HuggingFace `tokenizers` crate with `unstable_wasm` feature.
//!
//! Vocab is embedded at compile time via `include_bytes!` and lazy-initialized
//! into a `OnceLock<Tokenizer>` on first call. No runtime file I/O.
//!
//! Crate version + features are pinned in `Cargo.toml`. The Python training
//! pipeline uses the same underlying Rust crate (via `transformers`), so the
//! engine and training share tokenization implementation, not just compatibility.

use std::sync::OnceLock;
use tokenizers::{Tokenizer, TruncationParams, PaddingParams, PaddingStrategy};

/// BERT-base-uncased vocab, committed at `engine/assets/tokenizer.json`.
/// Updating this asset requires rebuilding all WASM artifacts.
static TOKENIZER_BYTES: &[u8] = include_bytes!(env!("TERN_TOKENIZER_PATH"));

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

/// Pad token id for BERT-base-uncased ([PAD]). Used to right-pad sequences
/// shorter than max_seq_len.
pub const PAD_TOKEN_ID: u32 = 0;

/// Maximum sequence length. Matches the value the model was trained at.
/// Inputs longer than this are truncated.
pub const MAX_SEQ_LEN: usize = 128;

fn get_tokenizer() -> &'static Tokenizer {
    TOKENIZER.get_or_init(|| {
        let mut tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES)
            .expect("embedded tokenizer.json is invalid — rebuild engine with a fresh asset");
        assert_eq!(tokenizer.token_to_id("[PAD]"), Some(PAD_TOKEN_ID));
        tokenizer.with_truncation(Some(TruncationParams {
            max_length: MAX_SEQ_LEN, ..Default::default()
        })).expect("invalid truncation settings");
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_SEQ_LEN),
            pad_id: PAD_TOKEN_ID, pad_token: "[PAD]".into(), ..Default::default()
        }));
        tokenizer
    })
}

/// Tokenize a string into token IDs.
///
/// - Adds [CLS] / [SEP] special tokens (BERT convention)
/// - Truncates to MAX_SEQ_LEN
/// - Pads with PAD_TOKEN_ID to MAX_SEQ_LEN
///
/// Returns a Vec<u32> of length MAX_SEQ_LEN.
pub fn tokenize(text: &str) -> Vec<u32> {
    encode(text).get_ids().to_vec()
}

pub fn encode(text: &str) -> tokenizers::Encoding {
    get_tokenizer().encode(ascii_digits(text), true).expect("tokenizer.encode failed")
}

/// Persian (۰-۹) and Arabic-Indic (٠-٩) digits → ASCII, so "۵ مگابیت" and
/// "5 مگابیت" get the same tokens. Done here rather than in tokenizer.json so
/// the tokenizer's SHA (pinned by checkpoints and packed models) is unchanged;
/// `bilingual_normalizer()` in training/distill/tokenization.py mirrors it.
fn ascii_digits(text: &str) -> String {
    text.chars().map(|c| match c {
        '۰'..='۹' => char::from(b'0' + (c as u32 - '۰' as u32) as u8),
        '٠'..='٩' => char::from(b'0' + (c as u32 - '٠' as u32) as u8),
        _ => c,
    }).collect()
}

pub fn vocab_size() -> usize {
    get_tokenizer().get_vocab_size(true)
}

pub fn sha256() -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(TOKENIZER_BYTES).iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_separator_and_padding() {
        let long = tokenize(&"hello ".repeat(200));
        assert_eq!(long.len(), MAX_SEQ_LEN);
        assert_eq!(long[MAX_SEQ_LEN - 1], get_tokenizer().token_to_id("[SEP]").unwrap());
        let short = tokenize("");
        assert_eq!(short[0], get_tokenizer().token_to_id("[CLS]").unwrap());
        assert_eq!(short[1], get_tokenizer().token_to_id("[SEP]").unwrap());
        assert!(short[2..].iter().all(|&id| id == PAD_TOKEN_ID));
    }

    #[test]
    fn persian_and_arabic_digits_tokenize_as_ascii() {
        assert_eq!(ascii_digits("۰۹ ٠٩ x5"), "09 09 x5");
        assert_eq!(tokenize("۵ مگابیت ٤٢"), tokenize("5 مگابیت 42"));
    }
}

/// Attention mask companion to `tokenize()`. Returns 1s for real tokens,
/// 0s for padding. Same length as the output of `tokenize()`.
pub fn attention_mask(token_ids: &[u32]) -> Vec<u8> {
    token_ids.iter().map(|&id| if id == PAD_TOKEN_ID { 0 } else { 1 }).collect()
}
