use pyo3::prelude::*;
use pyo3::types::PyBytes;
// use std::path::PathBuf;

mod transcribe;

// use polars::prelude::*;
// use tokenizers::tokenizer::Tokenizer;

// use indicatif::{ProgressBar, ParallelProgressIterator};
// use rayon::prelude::*;


const _MB_MASK: u8 = 0xC0;
const _MB_START: u8 = 0x80;


#[pyfunction]
#[pyo3(signature = (word, min_n=3, max_n=18))]
fn ngrams_bytes(word: &str, min_n: usize, max_n: usize, py: Python<'_>) -> PyResult<Vec<Py<PyBytes>>> {
    let wordbytes = word.as_bytes();
    let mut ngrams = Vec::new();
    let mut i = 0;
    while i < wordbytes.len() {
        if wordbytes[i] & _MB_MASK == _MB_START {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut n = 1;
        while j < wordbytes.len() && n <= max_n {
            j += 1;
            while j < wordbytes.len() && wordbytes[j] & _MB_MASK == _MB_START {
                j += 1;
            }
            if n >= min_n && !(n == 1 && (i == 0 || j == wordbytes.len())) {
                ngrams.push(wordbytes[i..j].to_vec());
            }
            n += 1;
        }
        i += 1;
    }
    let py_ngrams_bytes: Vec<Py<PyBytes>> = ngrams
        .iter()
        .map(|ngram| PyBytes::new(py, ngram).into())
        .collect();
    Ok(py_ngrams_bytes)
}

#[pyfunction]
#[pyo3(signature = (word, ns=vec![3, 6, 9, 12, 15, 18]))]
fn ngrams_bytes_specific(word: &str, ns: Vec<usize>, py: Python<'_>) -> PyResult<Vec<Py<PyBytes>>> {
    let wordbytes = word.as_bytes();
    let mut ngrams = Vec::new();
    let mut i = 0;
    // find max n to optimize early stopping
    let max_n = match ns.iter().max() {
        Some(&max) => max,
        None => return Ok(Vec::new()),
    };
    while i < wordbytes.len() {
        if wordbytes[i] & _MB_MASK == _MB_START {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut n = 1;
        while j < wordbytes.len() && n <= max_n {
            j += 1;
            while j < wordbytes.len() && wordbytes[j] & _MB_MASK == _MB_START {
                j += 1;
            }
            // only include n-grams of the specified sizes
            if ns.contains(&n) && !(n == 1 && (i == 0 || j == wordbytes.len())) {
                ngrams.push(wordbytes[i..j].to_vec());
            }
            n += 1;
        }
        i += 1;
    }
    let py_ngrams_bytes: Vec<Py<PyBytes>> = ngrams
        .iter()
        .map(|ngram| PyBytes::new(py, ngram).into())
        .collect();
    Ok(py_ngrams_bytes)
}

#[pyfunction]
#[pyo3(signature = (bytez))]
fn hash_bytes(bytez: Vec<u8>) -> PyResult<u32> {
    let mut h: u32 = 2166136261;
    for b in bytez {
        h = h ^ (b as u32);
        h = h.wrapping_mul(16777619);
    }
    Ok(h)
}


// #[pyfunction]
// #[pyo3(signature = (input_path, output_path, tokenizer_model_path))]
// fn add_sequence_lengths_rs(input_path: String, output_path: String, tokenizer_model_path: String) -> PyResult<()> {

//     println!("Rust: Loading tokenizer from {}...", tokenizer_model_path);
//     let tokenizer_path = PathBuf::from(tokenizer_model_path);
//     let tokenizer = Tokenizer::from_file(&tokenizer_path)
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load tokenizer: {}", e)))?;

//     println!("Rust: Fully configured tokenizer loaded.");

//     // load dataframe
//     println!("Rust: Loading DataFrame from {}...", input_path);
//     let file = std::fs::File::open(&input_path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
//     let mut df = IpcReader::new(file)
//         .memory_mapped(Some(PathBuf::from(input_path)))
//         .finish()
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to read IPC file: {}", e)))?;


//     println!("Rust: DataFrame loaded. Shape: {:?}", df.shape());

//     // get required columns
//     let metadata_series = df.column("metadata")
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Column 'metadata' not found: {}", e)))?
//         .str() // Ensure it's Utf8Chunked
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Column 'metadata' is not UTF8: {}", e)))?;

//     let original_text_series = df.column("original_text")
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Column 'original_text' not found: {}", e)))?
//         .str()
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Column 'original_text' is not UTF8: {}", e)))?;

//     // configure rayon thread pool
//     let num_threads = 64; // let's start with a more conservative number
//     println!("Rust: Configuring Rayon to use {} threads.", num_threads);
//     rayon::ThreadPoolBuilder::new()
//         .num_threads(num_threads)
//         .build_global()
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Failed to build Rayon global thread pool: {}", e)))?;

//     println!("Rust: Calculating sequence lengths (in parallel with {} threads)...", num_threads);

//     // process in chunks of 1 million rows
//     let chunk_size = 1_000_000;
//     let total_chunks = (df.height() + chunk_size - 1) / chunk_size; // Ceiling division
    
//     let mut all_lengths = Vec::with_capacity(df.height());
    
//     for chunk_idx in 0..total_chunks {
//         println!("Processing chunk {}/{}", chunk_idx + 1, total_chunks);
        
//         let start_idx = chunk_idx * chunk_size;
//         let end_idx = (start_idx + chunk_size).min(df.height());
        
//         let chunk_lengths: Vec<u32> = (start_idx..end_idx)
//             .into_par_iter()
//             .progress_with(ProgressBar::new((end_idx - start_idx) as u64))
//             .map(|i| {
//                 let meta_opt = metadata_series.get(i);
//                 let text_opt = original_text_series.get(i);
//                 let meta = meta_opt.unwrap_or("");
//                 let text = text_opt.unwrap_or("");
//                 let combined_text = format!("{} {}", meta, text);

//                 let encoding = tokenizer.encode(combined_text, true)
//                     .expect(&format!("Tokenization failed for row {}", i));

//                 encoding.get_ids().len() as u32
//             })
//             .collect();
            
//         // append to master list
//         all_lengths.extend(chunk_lengths);
//     }
    
//     // create series from the complete lengths vector
//     let lengths_series = Series::new("sequence_length".into(), all_lengths);
//     df.with_column(lengths_series)
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to add 'sequence_length' column: {}", e)))?; // `with_column` returns Result<&mut DataFrame, _>

//     println!("Rust: Added 'sequence_length' column. Final shape: {:?}", df.shape());

//     // write output dataFrame 
//     println!("Rust: Writing output DataFrame to {}...", output_path);
//     let output_file = std::fs::File::create(&output_path)
//          .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create output file: {}", e)))?;
//     // use IpcWriter for arrow format
//     IpcWriter::new(output_file)
//         .finish(&mut df)
//         .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to write IPC file: {}", e)))?;

//     println!("Rust: Successfully wrote output file.");
//     Ok(())
// }

// Python bindings for transcription functions
#[pyfunction]
fn transcribe_rs(text: &str) -> PyResult<String> {
    Ok(transcribe::transcribe_text(text))
}

#[pyfunction]
fn transcribe_batch_rs(texts: Vec<String>) -> PyResult<Vec<String>> {
    Ok(transcribe::transcribe_batch(texts))
}

#[pyfunction]
fn has_hanja(text: &str) -> PyResult<bool> {
    Ok(transcribe::contains_hanja(text))
}

#[pyfunction]
fn hanjain_rs(text: &str) -> PyResult<Vec<bool>> {
    Ok(text.chars().map(transcribe::is_hanja_char).collect())
}

#[pyfunction]
fn is_all_hanja(text: &str) -> PyResult<bool> {
    Ok(transcribe::is_all_hanja(text))
}

#[pymodule]
fn han2han_tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ngrams_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(ngrams_bytes_specific, m)?)?;
    m.add_function(wrap_pyfunction!(hash_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(transcribe_rs, m)?)?;
    m.add_function(wrap_pyfunction!(transcribe_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(has_hanja, m)?)?;
    m.add_function(wrap_pyfunction!(hanjain_rs, m)?)?;
    m.add_function(wrap_pyfunction!(is_all_hanja, m)?)?;
    // m.add_function(wrap_pyfunction!(add_sequence_lengths_rs, m)?)?;
    Ok(())
}