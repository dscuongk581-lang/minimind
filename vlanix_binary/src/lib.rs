//! Fixed byte interface and tied FP32 embedding reference. No I/O or unsafe.
#![forbid(unsafe_code)]

pub const VOCAB: usize = 265;
pub const WIDTH: usize = 256;
pub const BOS: u16 = 1;
pub const EOS: u16 = 2;
const BYTE_BASE: u16 = 9;
const MAGIC: &[u8; 8] = b"VLXEMB01";
const HEADER: usize = 16;

#[derive(Debug, PartialEq, Eq)]
pub enum Error { Frame, Token, Shape, NonFinite, Format }

/// All 256 byte values, including invalid UTF-8, have independent fixed IDs.
/// Control codes are out of band in the u16 ID stream, never raw byte values.
pub fn encode(bytes: &[u8]) -> Vec<u16> {
    let mut ids = Vec::new();
    ids.push(BOS);
    ids.extend(bytes.iter().map(|&b| u16::from(b) + BYTE_BASE));
    ids.push(EOS);
    ids
}

pub fn decode(ids: &[u16]) -> Result<Vec<u8>, Error> {
    if ids.len() < 2 || ids.first() != Some(&BOS) || ids.last() != Some(&EOS) {
        return Err(Error::Frame);
    }
    ids[1..ids.len()-1].iter().map(|&id| {
        id.checked_sub(BYTE_BASE).and_then(|v| u8::try_from(v).ok()).ok_or(Error::Token)
    }).collect()
}

/// Exactly one contiguous row-major matrix, used for lookup and output logits.
/// Logits are scores, not inverse embeddings or a guarantee of byte recovery.
pub struct Embedding { weights: Vec<f32> }

impl Embedding {
    pub fn new(weights: Vec<f32>) -> Result<Self, Error> {
        if weights.len() != VOCAB * WIDTH { return Err(Error::Shape); }
        if !weights.iter().all(|w| w.is_finite()) { return Err(Error::NonFinite); }
        Ok(Self { weights })
    }

    pub fn row(&self, id: u16) -> Result<&[f32], Error> {
        let i = usize::from(id);
        if i >= VOCAB { return Err(Error::Token); }
        Ok(&self.weights[i*WIDTH..(i+1)*WIDTH])
    }

    pub fn set(&mut self, id: u16, col: usize, value: f32) -> Result<(), Error> {
        if usize::from(id) >= VOCAB || col >= WIDTH { return Err(Error::Shape); }
        if !value.is_finite() { return Err(Error::NonFinite); }
        self.weights[usize::from(id)*WIDTH+col] = value;
        Ok(())
    }

    pub fn logits(&self, h: &[f32; WIDTH]) -> Result<Vec<f32>, Error> {
        if !h.iter().all(|x| x.is_finite()) { return Err(Error::NonFinite); }
        self.weights.chunks_exact(WIDTH).map(|row| {
            let score = row.iter().zip(h.iter()).fold(0.0f32, |sum, (&w,&x)| sum+w*x);
            if score.is_finite() { Ok(score) } else { Err(Error::NonFinite) }
        }).collect()
    }

    /// Portable reference snapshot, not a full organism checkpoint or a signature.
    pub fn snapshot(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + VOCAB*WIDTH*4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(VOCAB as u32).to_le_bytes());
        out.extend_from_slice(&(WIDTH as u32).to_le_bytes());
        for &w in &self.weights { out.extend_from_slice(&w.to_le_bytes()); }
        out
    }

    pub fn restore(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != HEADER + VOCAB*WIDTH*4 ||
            &bytes[..8] != MAGIC ||
            bytes[8..12] != (VOCAB as u32).to_le_bytes() ||
            bytes[12..16] != (WIDTH as u32).to_le_bytes() {
            return Err(Error::Format);
        }
        Self::new(bytes[HEADER..].chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect())
    }
}
