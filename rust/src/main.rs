//! Phase-SNN inference CLI
//!
//! Usage:
//!   phase_snn_inference --weights weights.json --prompt "The history of"
//!   phase_snn_inference --weights weights.json --interactive

use phase_snn::PhaseLM;
use std::io::{self, BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let weights_path = args.iter().position(|a| a == "--weights")
        .and_then(|i| args.get(i+1))
        .map(|s| s.as_str())
        .unwrap_or("weights.json");

    let model = match PhaseLM::load(weights_path) {
        Ok(m)  => m,
        Err(e) => { eprintln!("Failed to load: {}", e); return; }
    };

    if args.contains(&"--interactive".to_string()) {
        println!("Phase-SNN interactive mode. Type a prompt, Enter to generate.");
        let stdin = io::stdin();
        loop {
            print!("> ");
            io::stdout().flush().unwrap();
            let mut line = String::new();
            if stdin.lock().read_line(&mut line).is_err() { break; }
            let line = line.trim();
            if line.is_empty() { continue; }
            if line == "quit" { break; }
            // Simple word tokenisation for demo
            // Real deployment would use the saved vocab
            let tokens: Vec<u32> = vec![2];  // <bos>
            let generated = model.generate(&tokens, 50, 0.8);
            println!("Generated: {:?}", generated);
        }
    } else {
        // Benchmark mode
        let tokens = vec![2u32, 10, 42, 7, 15, 3, 8];
        let start  = std::time::Instant::now();
        let iters  = 1000;
        for _ in 0..iters {
            let _ = model.predict_next(&tokens);
        }
        let elapsed = start.elapsed();
        println!("Inference: {:.3}ms/query ({} queries)",
                 elapsed.as_secs_f64() * 1000.0 / iters as f64, iters);
    }
}
