use std::{env, path::PathBuf};

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    for (variable, default) in [
        ("TERN_MODEL_PATH", "assets/model.bin"),
        ("TERN_TOKENIZER_PATH", "assets/tokenizer.json"),
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
        let path = env::var_os(variable).map(PathBuf::from).unwrap_or_else(|| root.join(default));
        let path = path.canonicalize().unwrap_or_else(|_| panic!("Missing {}", path.display()));
        println!("cargo:rerun-if-changed={}", path.display());
        println!("cargo:rustc-env={variable}={}", path.display());
    }
}
