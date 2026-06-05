use hashbrown::HashMap;
use once_cell::sync::Lazy;
use std::collections::BTreeMap;

// Unicode Hanja ranges
const HANJA_RANGES: &[(u32, u32)] = &[
    (0x4E00, 0x9FFF),   // CJK Unified Ideographs
    (0x3400, 0x4DBF),   // CJK Extension A
    (0x20000, 0x2A6DF), // CJK Extension B
    (0x2A700, 0x2B73F), // CJK Extension C
    (0x2B740, 0x2B81F), // CJK Extension D
    (0x2B820, 0x2CEAF), // CJK Extension E
    (0x2CEB0, 0x2EBEF), // CJK Extension F
    (0xF900, 0xFAFF),   // CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), // CJK Compatibility Supplement
];

// Doeum law transformations
static DOEUM_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("녀", "여");
    map.insert("념", "염");
    map.insert("뇨", "요");
    map.insert("뉴", "유");
    map.insert("니", "이");
    map.insert("랴", "야");
    map.insert("려", "여");
    map.insert("례", "예");
    map.insert("료", "요");
    map.insert("류", "유");
    map.insert("리", "이");
    map.insert("림", "임");
    map.insert("라", "나");
    map.insert("래", "내");
    map.insert("로", "노");
    map.insert("뢰", "뇌");
    map.insert("루", "누");
    map.insert("르", "느");
    map
});

// Global dictionary storage
static HANJA_DICT: Lazy<HashMap<String, String>> = Lazy::new(|| {
    let mut dict = HashMap::new();

    // Load dictionaries - embedded at compile time for speed
    let dicts = [
        include_str!("../python/han2han_tools/dict/dic0.txt"),
        include_str!("../python/han2han_tools/dict/dic4.txt"),
        include_str!("../python/han2han_tools/dict/dic1.txt"),
    ];

    for dict_content in dicts {
        for line in dict_content.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
                dict.insert(parts[0].to_string(), parts[1].to_string());
            }
        }
    }

    dict
});

// Multi-character matches sorted by length (longest first)
static MULTI_CHAR_DICT: Lazy<BTreeMap<usize, Vec<(String, String)>>> = Lazy::new(|| {
    let mut by_length: BTreeMap<usize, Vec<(String, String)>> = BTreeMap::new();

    for (hanja, hangul) in HANJA_DICT.iter() {
        let len = hanja.chars().count();
        if len > 1 {
            by_length.entry(len).or_insert_with(Vec::new)
                .push((hanja.clone(), hangul.clone()));
        }
    }

    by_length
});

// Quick check if char is Hanja
#[inline]
pub fn is_hanja_char(ch: char) -> bool {
    let code = ch as u32;
    HANJA_RANGES.iter().any(|&(start, end)| code >= start && code <= end)
}

// Check if string contains any Hanja
pub fn contains_hanja(text: &str) -> bool {
    text.chars().any(is_hanja_char)
}

// Check if string is all Hanja characters
pub fn is_all_hanja(text: &str) -> bool {
    !text.is_empty() && text.chars().all(is_hanja_char)
}

// Main transcription function
pub fn transcribe_text(text: &str) -> String {
    // Quick return if no Hanja
    if !contains_hanja(text) {
        return text.to_string();
    }

    let mut result = String::with_capacity(text.len() * 3); // Hangul typically longer
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if !is_hanja_char(ch) {
            result.push(ch);
            i += 1;
            continue;
        }

        let mut matched = false;

        // Try multi-character matches first (longest to shortest)
        for (&len, patterns) in MULTI_CHAR_DICT.iter().rev() {
            if i + len > chars.len() {
                continue;
            }

            let candidate: String = chars[i..i + len].iter().collect();

            if let Some(entry) = patterns.iter().find(|(h, _)| h == &candidate) {
                result.push_str(&entry.1);
                i += len;
                matched = true;
                break;
            }
        }

        if !matched {
            // Single character lookup
            let ch_str = ch.to_string();
            if let Some(hangul) = HANJA_DICT.get(&ch_str) {
                // Apply doeum law for single characters
                if let Some(transformed) = DOEUM_MAP.get(hangul.as_str()) {
                    result.push_str(transformed);
                } else {
                    result.push_str(hangul);
                }
            } else {
                // No translation found, keep original
                result.push(ch);
            }
            i += 1;
        }
    }

    result
}

// Batch processing for multiple strings
pub fn transcribe_batch(texts: Vec<String>) -> Vec<String> {
    texts.into_iter()
        .map(|text| transcribe_text(&text))
        .collect()
}