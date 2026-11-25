//! Bounded offline test harness; not an external input service.
#![forbid(unsafe_code)]
use std::{env, fs, io, path::Path};
use vlanix_binary::{decode, encode, Embedding, VOCAB, WIDTH};

const BATCH: usize = 8;

fn read_exact(path: &Path, len: usize) -> io::Result<Vec<u8>> {
    use std::io::Read;
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take((len+1) as u64).read_to_end(&mut bytes)?;
    if bytes.len()!=len { return Err(io::Error::new(io::ErrorKind::InvalidData,"length")); }
    Ok(bytes)
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn project(e: &Embedding, h: &[[f32; WIDTH]]) -> Vec<u8> {
    h.iter().flat_map(|v| floats(&e.logits(v).expect("finite logits"))).collect()
}

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len()!=3 { return Err(io::Error::new(io::ErrorKind::InvalidInput,"input and output directories")); }
    let input=Path::new(&args[1]); let out=Path::new(&args[2]);
    let snapshot=read_exact(&input.join("embedding.bin"),16+VOCAB*WIDTH*4)?;
    let mut e=Embedding::restore(&snapshot).expect("valid embedding");
    let hidden=read_exact(&input.join("hidden.bin"),BATCH*WIDTH*4)?;
    let h: Vec<[f32;WIDTH]> = hidden.chunks_exact(WIDTH*4).map(|row| {
        std::array::from_fn(|i| f32::from_le_bytes(row[4*i..4*i+4].try_into().unwrap()))
    }).collect();
    let payload=read_exact(&input.join("payload.bin"),256)?;
    let ids=encode(&payload);
    let decoded=decode(&ids).expect("valid frame");
    let id_bytes: Vec<u8>=ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let rows: Vec<u8>=(0..VOCAB).flat_map(|id| floats(e.row(id as u16).unwrap())).collect();
    fs::write(out.join("embedding.bin"),e.snapshot())?;
    fs::write(out.join("payload.bin"),decoded)?;
    fs::write(out.join("ids.bin"),id_bytes)?;
    fs::write(out.join("rows.bin"),rows)?;
    fs::write(out.join("logits.bin"),project(&e,&h))?;
    let next=e.row(9).unwrap()[0]+0.25;
    e.set(9,0,next).expect("valid update");
    fs::write(out.join("mutated.bin"),e.snapshot())?;
    fs::write(out.join("mutated_row.bin"),floats(e.row(9).unwrap()))?;
    fs::write(out.join("mutated_logits.bin"),project(&e,&h))?;
    Ok(())
}
