//! The models are compiled in with `include_bytes!`: a missing file says how to fetch it, not just that it is absent.

use std::path::Path;

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
}
