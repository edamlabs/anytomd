//! The models are compiled in with `include_bytes!`: a missing file says how to fetch it, not just that it is absent.
//! On macOS the static ONNX Runtime calls `__isPlatformVersionAtLeast` (clang's `@available`); a Python extension links
//! with `-undefined dynamic_lookup`, so the symbol stays open until `dlopen` fails. compiler-rt's builtins close it.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const EMBEDDED: [&str; 6] = [
    "pp-ocrv6_tiny_det.onnx",
    "latin_pp-ocrv5_mobile_rec.onnx",
    "ppocrv5_latin_dict.txt",
    "cyrillic_pp-ocrv5_mobile_rec.onnx",
    "ppocrv5_cyrillic_dict.txt",
    "pp-lcnet_x1_0_doc_ori.onnx",
];

fn main() {
    let missing: Vec<&str> = EMBEDDED
        .into_iter()
        .filter(|name| !Path::new("models").join(name).is_file())
        .collect();
    assert!(
        missing.is_empty(),
        "embedded OCR models missing from crates/oar/models: {missing:?}; run `python3 tools/get_oar_models.py --embedded`"
    );
    EMBEDDED
        .iter()
        .for_each(|name| println!("cargo::rerun-if-changed=models/{name}"));
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        link_compiler_rt();
    }
}

fn link_compiler_rt() {
    let output = Command::new("xcrun")
        .args(["clang", "--print-file-name=libclang_rt.osx.a"])
        .output()
        .expect("xcrun runs on a macOS build host");
    let archive = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    assert!(
        archive.is_file(),
        "libclang_rt.osx.a not found ({}): install the Xcode command line tools",
        archive.display()
    );
    let directory = archive.parent().expect("an archive lives in a directory");
    println!("cargo::rustc-link-search=native={}", directory.display());
    println!("cargo::rustc-link-lib=static=clang_rt.osx");
}
