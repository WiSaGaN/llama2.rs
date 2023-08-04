use std::mem::MaybeUninit;
use std::io::Read;
use std::time::SystemTime;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Config {
    dim: i32, // transformer dimension
    hidden_dim: i32, // for ffn layers
    n_layers: i32, // number of layers
    n_heads: i32, // number of query heads
    n_kv_heads: i32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: i32, // vocabulary size, usually 256 (byte-level)
    seq_len: i32, // max sequence length
}

#[derive(Copy, Clone, Debug)]
struct ConfigUsize {
    dim: usize, // transformer dimension
    hidden_dim: usize, // for ffn layers
    n_layers: usize, // number of layers
    n_heads: usize, // number of query heads
    n_kv_heads: usize, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: usize, // vocabulary size, usually 256 (byte-level)
    seq_len: usize, // max sequence length
}

impl From<Config> for ConfigUsize {
    fn from(c: Config) -> Self {
        ConfigUsize {
            dim: c.dim as usize,
            hidden_dim: c.hidden_dim as usize,
            n_layers: c.n_layers as usize,
            n_heads: c.n_heads as usize,
            n_kv_heads: c.n_kv_heads as usize,
            vocab_size: c.vocab_size as usize,
            seq_len: c.seq_len as usize,
        }
    }
}

#[derive(Clone, Default, Debug)]
struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>,    // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
}

fn read_into<T: Copy>(buffer: &mut Vec<T>, reader: &mut dyn Read, size: usize) -> Result<(), std::io::Error> {
    buffer.reserve(size);
    unsafe {
        let ptr = buffer.as_mut_ptr();
        let byte_buffer = std::slice::from_raw_parts_mut(ptr as *mut u8, size * std::mem::size_of::<T>());
        reader.read_exact(byte_buffer)?;
        buffer.set_len(size);
    }
    Ok(())
}

impl TransformerWeights {
    pub fn checkpoint_init_weights(file: &mut dyn Read, config: &ConfigUsize) -> Result<Self, std::io::Error> {
        let mut weights = Self::default();
        read_into(&mut weights.token_embedding_table, file, config.vocab_size * config.dim)?;
        read_into(&mut weights.rms_att_weight, file, config.n_layers * config.dim)?;
        read_into(&mut weights.wq, file, config.n_layers * config.dim * config.dim)?;
        read_into(&mut weights.wk, file, config.n_layers * config.dim * config.dim)?;
        read_into(&mut weights.wv, file, config.n_layers * config.dim * config.dim)?;
        read_into(&mut weights.wo, file, config.n_layers * config.dim * config.dim)?;
        read_into(&mut weights.rms_ffn_weight, file, config.n_layers * config.dim)?;
        read_into(&mut weights.w1, file, config.n_layers * config.hidden_dim * config.dim)?;
        read_into(&mut weights.w2, file, config.n_layers * config.dim * config.hidden_dim)?;
        read_into(&mut weights.w3, file, config.n_layers * config.hidden_dim * config.dim)?;
        read_into(&mut weights.rms_final_weight, file, config.dim)?;
        let head_size = config.dim / config.n_heads;
        weights.freq_cis_real = vec![0.0; config.seq_len * config.dim / 2];
        unsafe {
            let ptr = weights.freq_cis_real.as_mut_ptr();
            let byte_buffer = std::slice::from_raw_parts_mut(ptr as *mut u8, (config.seq_len * head_size / 2) * std::mem::size_of::<f32>());
            file.read_exact(byte_buffer)?;
        }
        weights.freq_cis_imag = vec![0.0; config.seq_len * config.dim / 2];
        unsafe {
            let ptr = weights.freq_cis_imag.as_mut_ptr();
            let byte_buffer = std::slice::from_raw_parts_mut(ptr as *mut u8, (config.seq_len * head_size / 2) * std::mem::size_of::<f32>());
            file.read_exact(byte_buffer)?;
        }
        Ok(weights)
    }
}

struct RunState {
    // current wave of activations
    x: Vec<f32>, // activation at current time stamp (dim,)
    xb: Vec<f32>, // same, but inside a residual branch (dim,)
    xb2: Vec<f32>, // an additional buffer just for convenience (dim,)
    hb: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>, // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>, // query (dim,)
    k: Vec<f32>, // key (dim,)
    v: Vec<f32>, // value (dim,)
    att: Vec<f32>, // buffer for scores/attention values (seq_len,)
    logits: Vec<f32>, // output logits
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    pub fn new(config: &ConfigUsize) -> Self {
        let x = vec![0.0; config.dim];
        let xb = vec![0.0; config.dim];
        let xb2 = vec![0.0; config.dim];
        let hb = vec![0.0; config.hidden_dim];
        let hb2 = vec![0.0; config.hidden_dim];
        let q = vec![0.0; config.dim];
        let k = vec![0.0; config.dim];
        let v = vec![0.0; config.dim];
        let att = vec![0.0; config.seq_len];
        let logits = vec![0.0; config.vocab_size];
        let key_cache = vec![0.0; config.n_layers * config.seq_len * config.dim];
        let value_cache = vec![0.0; config.n_layers * config.seq_len * config.dim];
        RunState {
            x,
            xb,
            xb2,
            hb,
            hb2,
            q,
            k,
            v,
            att,
            logits,
            key_cache,
            value_cache,
        }
    }
}

fn accum(a: &mut[f32], b: &[f32], size: usize) {
    for i in 0..size {
        a[i] += b[i]
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32], size: usize) {
    // calculate sum of squares
    let mut ss = 0.0;
    for j in 0..size {
        ss += x[j] * x[j];
    }
    ss /= size as f32;
    ss += 1e-5;
    ss = 1.0 / ss.sqrt();
    // normalize and scale
    for j in 0..size {
        o[j] = weight[j] * (ss * x[j]);
    }
}

fn softmax(x: &mut [f32]) {
    assert!(!x.is_empty());
    if x.len() == 1 {
        x[0] = 1.0;
        return;
    }
    // find max value (for numerical stability)
    let max_val = x.iter().copied().max_by(|l, r| l.total_cmp(r)).unwrap_or(x[0]);
    // e^x
    x.iter_mut().for_each(|v| *v = (*v - max_val).exp());
    // normalize
    let sum: f32 = x.iter().copied().sum();
    x.iter_mut().for_each(|v| *v /= sum);
    assert!(x.iter().copied().sum::<f32>() < 1.01);
    assert!(x.iter().copied().sum::<f32>() > 0.99);
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    assert!(x.len() >= n);
    assert!(w.len() >= d * n);
    for i in 0..d {
        let mut val = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn transformer(token: i32, pos: usize, p: &ConfigUsize, s: &mut RunState, w: &TransformerWeights) {
    // a few convenice variables
    let x = &mut s.x;
    let token = token as usize;
    let dim = p.dim;
    let hidden_dim = p.hidden_dim;
    let head_size = dim / p.n_heads;

    // copy the token embedding into x
    let content_row = &w.token_embedding_table[token * dim..token * dim + dim];
    x.copy_from_slice(content_row);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = &w.freq_cis_real[(pos * head_size / 2)..];
    let freq_cis_imag_row = &w.freq_cis_imag[(pos * head_size / 2)..];

    // forward all the layers
    for l in 0..p.n_layers {
        // attention rmsnorm
        rmsnorm(&mut s.xb, x, &w.rms_att_weight[l*dim..], dim);

        // qkv matmuls for this position
        matmul(&mut s.q, &s.xb, &w.wq[l*dim*dim..], dim, dim);
        matmul(&mut s.k, &s.xb, &w.wk[l*dim*dim..], dim, dim);
        matmul(&mut s.v, &s.xb, &w.wv[l*dim*dim..], dim, dim);

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..p.n_heads {
            // get the q and k vectors for this head
            let q = &mut s.q[h * head_size..h * head_size + head_size];
            let k = &mut s.k[h * head_size..h * head_size + head_size];
            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i]     = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i]     = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }
        // save key,value at this time step (pos) to our kv cache
        let loff = l * p.seq_len * dim; // kv cache layer offset for convenience
        let key_cache_row = &mut s.key_cache[(loff + pos * dim)..];
        let value_cache_row = &mut s.value_cache[(loff + pos * dim)..];
        key_cache_row[..dim].copy_from_slice(&s.k[..dim]);
        value_cache_row[..dim].copy_from_slice(&s.v[..dim]);

        // multihead attention. iterate over all heads
        for h in 0..p.n_heads {
            // get the query vector for this head
            let q = &s.q[(h * head_size)..];
            // iterate over all timesteps, including the current one
            for t in 0..pos + 1 {
                // get the key vector for this head and at this timestep
                let k = &s.key_cache[(loff + t * dim + h * head_size)..];
                // calculate the attention score as the dot product of q and k
                let mut score = 0.0;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                s.att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(&mut s.att[..pos + 1]);

            // weighted sum of the values, store back into xb
            for i in 0..head_size {
                let mut val = 0.0;
                for t in 0..pos + 1 {
                    val += s.att[t] * s.value_cache[(loff + t * dim + h * head_size) + i]; // note bad locality
                }
                s.xb[(h * head_size) + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(&mut s.xb2, &s.xb, &w.wo[(l*dim*dim)..], dim, dim);

        // residual connection back into x
        accum(x, &s.xb2, dim);

        // ffn rmsnorm
        rmsnorm(&mut s.xb, x, &w.rms_ffn_weight[(l*dim)..], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(&mut s.hb, &s.xb, &w.w1[(l*dim*hidden_dim)..], dim, hidden_dim);
        matmul(&mut s.hb2, &s.xb, &w.w3[(l*dim*hidden_dim)..], dim, hidden_dim);

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for i in 0..hidden_dim {
            s.hb[i] *= 1.0 / (1.0 + (-s.hb[i]).exp());
        }

        // elementwise multiply with w3(x)
        for i in 0..hidden_dim {
            s.hb[i] *= s.hb2[i];
        }

        // final matmul to get the output of the ffn
        matmul(&mut s.xb, &s.hb, &w.w2[(l*dim*hidden_dim)..], hidden_dim, dim);

        // residual connection
        accum(x, &s.xb, dim);
    }

    // final rmsnorm
    let x_copy = x.to_owned();
    rmsnorm(x, &x_copy, &w.rms_final_weight, dim);

    // classifier into logits
    matmul(&mut s.logits, x, &w.token_embedding_table, p.dim, p.vocab_size);
}

fn sample(weights: &[f32], rng: &mut Lcg) -> usize {
    let r: f32 = rng.next() as f32 / Lcg::M as f32;
    let mut cdf = 0.0;
    for (i, &weight) in weights.iter().enumerate() {
        cdf += weight;
        if r < cdf {
            return i;
        }
    }
    weights.len() - 1
}

// Simple rand to avoid including rand dependency
struct Lcg {
    state: u64,
}

impl Lcg {
    pub const M: u64 = 1u64 << 32;
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        // Constants from "Numerical Recipes"
        const A: u64 = 1664525;
        const C: u64 = 1013904223;

        self.state = (A.wrapping_mul(self.state) + C) % Self::M;
        self.state
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let (checkpoint, temperature, seed) = match args.as_slice() {
        [_, checkpoint] => {
            let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as u64;
            (checkpoint, 0.9, seed)
        }
        [_, checkpoint, temperature] => {
            let temperature = temperature.parse()?;
            let seed = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() as u64;
            (checkpoint, temperature, seed)
        }
        [_, checkpoint, temperature, seed] => {
            let temperature = temperature.parse()?;
            let seed = seed.parse()?;
            (checkpoint, temperature, seed)
        }
        _ => {
            let s = format!("Usage: {} <checkpoint_file> [temperature] [seed]", args[0]);
            return Err(s.into());
        }
    };
    let mut rng = Lcg::new(seed);

    let mut file = std::fs::File::open(checkpoint)?;

    let config: Config = unsafe {
        let mut config = MaybeUninit::uninit();
        let buffer = std::slice::from_raw_parts_mut(config.as_mut_ptr() as *mut u8, std::mem::size_of::<Config>());
        file.read_exact(buffer)?;
        config.assume_init()
    };
    let config: ConfigUsize = ConfigUsize::from(config);

    let weights = TransformerWeights::checkpoint_init_weights(&mut file, &config)?;
    let mut state = RunState::new(&config);

    // the current position we are in
    let mut next;
    let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
    let mut pos = 0;

    while pos < config.seq_len {
        transformer(token, pos, &config, &mut state, &weights);

        // sample the next token
        if temperature == 0.0 {
            next = state.logits.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index)
                .unwrap_or(0) as i32;
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size {
                state.logits[q] /= temperature;
            }
            // apply softmax to the logits to get the probabilities for next token
            softmax(&mut state.logits);
            // we now want to sample from this distribution to get the next token

            next = sample(&state.logits, &mut rng) as i32;
        }
        println!("{next}");
        token = next;
        pos += 1;
    }

    Ok(())
}
