use vlanix_binary::*;

#[test]
fn all_bytes_and_invalid_utf8_roundtrip() {
    let bytes: Vec<u8> = (0..=255).collect();
    let ids = encode(&bytes);
    assert_eq!(ids.len(), 258);
    assert_eq!(ids[0], BOS);
    assert_eq!(ids[1], 9);
    assert_eq!(ids[256], 264);
    assert_eq!(ids[257], EOS);
    assert_eq!(decode(&ids).unwrap(), bytes);
    assert_eq!(decode(&encode(&[])).unwrap(), Vec::<u8>::new());
}

#[test]
fn strict_framing() {
    for ids in [vec![], vec![BOS], vec![9, EOS],
        vec![BOS, EOS, EOS], vec![BOS, 0, EOS],
        vec![BOS, 265, EOS], vec![BOS, BOS, EOS]] {
        assert!(decode(&ids).is_err());
    }
}

fn fixture() -> Embedding {
    let mut w = vec![0.0; VOCAB * WIDTH];
    w[9 * WIDTH] = 2.0;
    w[10 * WIDTH] = -3.0;
    w[264 * WIDTH + 1] = 4.0;
    Embedding::new(w).unwrap()
}

#[test]
fn tied_projection_uses_same_matrix() {
    let mut e = fixture();
    let mut h = [0.0; WIDTH];
    h[0] = 3.0; h[1] = 2.0;
    assert_eq!(e.row(9).unwrap()[0], 2.0);
    let y = e.logits(&h).unwrap();
    assert_eq!((y[9], y[10], y[264]), (6.0, -9.0, 8.0));
    e.set(9, 0, 5.0).unwrap();
    assert_eq!(e.row(9).unwrap()[0], 5.0);
    assert_eq!(e.logits(&h).unwrap()[9], 15.0);
    assert!(e.row(265).is_err());
    assert!(e.set(9, WIDTH, 0.0).is_err());
    assert!(e.set(9, 0, f32::NAN).is_err());
}

#[test]
fn snapshot_exact_and_canonical() {
    let e = fixture();
    let s = e.snapshot();
    assert_eq!(s.len(), 16 + VOCAB * WIDTH * 4);
    assert_eq!(&s[0..8], b"VLXEMB01");
    assert_eq!(&s[8..12], &(265u32).to_le_bytes());
    assert_eq!(&s[12..16], &(256u32).to_le_bytes());
    let offset = 16 + 9 * WIDTH * 4;
    assert_eq!(&s[offset..offset+4], &2.0f32.to_le_bytes());
    let restored = Embedding::restore(&s).unwrap();
    assert_eq!(restored.snapshot(), s);
}

#[test]
fn snapshot_rejects_malformed() {
    let s = fixture().snapshot();
    assert!(Embedding::restore(&s[..s.len()-1]).is_err());
    let mut trailing = s.clone(); trailing.push(0);
    assert!(Embedding::restore(&trailing).is_err());
    let mut bad = s.clone(); bad[0] ^= 1;
    assert!(Embedding::restore(&bad).is_err());
    let mut bad = s.clone(); bad[8] ^= 1;
    assert!(Embedding::restore(&bad).is_err());
    let mut bad = s.clone(); bad[12] ^= 1;
    assert!(Embedding::restore(&bad).is_err());
    let mut bad = s; bad[16..20].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(Embedding::restore(&bad).is_err());
    assert!(Embedding::new(vec![]).is_err());
}

#[test]
fn nonfinite_math_rejected() {
    let e = fixture();
    let mut h = [0.0; WIDTH]; h[0] = f32::NAN;
    assert!(e.logits(&h).is_err());
    h[0] = f32::MAX;
    assert!(e.logits(&h).is_err());
}
