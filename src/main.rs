use std::sync::Arc;

use parking_lot::RwLock;
use snarkvm::{
    algorithms::fft::EvaluationDomain,
    create_scalar_bases,
    prelude::{TestRng, ToBytes, Uniform},
    Fr, G1Affine, G1Projective, Zero,
};

pub use snarkvm;
pub mod cuda;
pub mod fft;
pub mod opencl;
use snarkvm::prelude::*;

use crate::fft::fft_opencl;
fn main() {}

#[test]
fn test_fft() {
    let rng = &mut TestRng::default();
    let log_d = 9;
    let d = 1 << log_d;
    let domain = EvaluationDomain::<Fr>::new(d).unwrap();

    let log_n = log_d;
    let omega = domain.group_gen;
    let expected = (0..d).map(|_| Fr::rand(rng)).collect::<Vec<_>>();
    ////////////////////////////////////////////////
    // cpu part
    let mut cpu_fft = expected.clone();
    fft_in_place(&mut cpu_fft, omega, log_n);

    ////////////////////////////////////////////////
    // gpu part
    let gpu_fft = expected.clone();
    let gpu_fft = Arc::new(RwLock::new(gpu_fft));
    // fft_in_place(&mut gpu_fft, omega, log_n);
    fft_opencl(gpu_fft.clone(), omega, log_n).unwrap();
    let gpu_fft = gpu_fft.write();
    // .as_slice();
    ////////////////////////////////////////////////

    assert_eq!(&cpu_fft, gpu_fft.as_slice());
}

fn fft_in_place(a: &mut [Fr], omega: Fr, log_n: u32) {
    #[inline]
    pub(crate) fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }
    // use core::convert::TryFrom;
    let n =
        u32::try_from(a.len()).expect("cannot perform FFTs larger on vectors of len > (1 << 32)");
    assert_eq!(n, 1 << log_n);

    // swap coefficients in place
    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _i in 1..=log_n {
        // w_m is 2^i-th root of unity
        let w_m = omega.pow([(n / (2 * m)) as u64]);

        let mut k = 0;
        while k < n {
            // w = w_m^j at the start of every loop iteration
            let mut w = Fr::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t *= w;
                let mut tmp = a[(k + j) as usize];
                tmp -= t;
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize] += t;
                w *= &w_m;
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn ifft_in_place(a: &mut [Fr], omega: Fr, log_n: u32) {
    fft_in_place(a, omega.inverse().unwrap(), log_n);
    let domain_size_inv = Fr::from(a.len() as u64).inverse().unwrap();
    for coeff in a.iter_mut() {
        *coeff *= domain_size_inv;
    }
}

#[test]
pub fn test_fft_fr() {
    let rng = &mut TestRng::default();
    let log_d = 9;
    let d = 1 << log_d;
    let domain = EvaluationDomain::<Fr>::new(d).unwrap();

    ////////////////////////////////////////////////
    // cpu part
    let omega = domain.group_gen;
    let log_n = log_d;
    let expected = (0..d).map(|_| Fr::rand(rng)).collect::<Vec<_>>();

    let mut cpu_fft = expected.clone();
    fft_in_place(&mut cpu_fft, omega, log_n);

    ////////////////////////////////////////////////

    let mut v = vec![];
    for _ in 0..d {
        v.push(Fr::rand(rng));
    }
    // Fill up with zeros.
    v.resize(domain.size(), Fr::zero());
    let mut v2 = v.clone();

    domain.ifft_in_place(&mut v2);
    domain.fft_in_place(&mut v2);
    assert_eq!(v, v2, "ifft(fft(.)) != iden");

    domain.fft_in_place(&mut v2);
    domain.ifft_in_place(&mut v2);
    assert_eq!(v, v2, "fft(ifft(.)) != iden");

    domain.coset_ifft_in_place(&mut v2);
    domain.coset_fft_in_place(&mut v2);
    assert_eq!(v, v2, "coset_fft(coset_ifft(.)) != iden");

    domain.coset_fft_in_place(&mut v2);
    domain.coset_ifft_in_place(&mut v2);
    assert_eq!(v, v2, "coset_ifft(coset_fft(.)) != iden");
}

#[test]
pub fn test_fft_g1_projective() {
    let rng = &mut TestRng::default();
    let log_d = 9;
    let d = 1 << log_d;
    let domain = EvaluationDomain::<Fr>::new(d).unwrap();

    let mut v = vec![];
    for _ in 0..d {
        v.push(G1Projective::rand(rng));
    }
    // Fill up with zeros.
    v.resize(domain.size(), G1Projective::zero());
    let mut v2 = v.clone();

    domain.ifft_in_place(&mut v2);
    domain.fft_in_place(&mut v2);
    assert_eq!(v, v2, "ifft(fft(.)) != iden");

    domain.fft_in_place(&mut v2);
    domain.ifft_in_place(&mut v2);
    assert_eq!(v, v2, "fft(ifft(.)) != iden");

    domain.coset_ifft_in_place(&mut v2);
    domain.coset_fft_in_place(&mut v2);
    assert_eq!(v, v2, "coset_fft(coset_ifft(.)) != iden");

    domain.coset_fft_in_place(&mut v2);
    domain.coset_ifft_in_place(&mut v2);
    assert_eq!(v, v2, "coset_ifft(coset_fft(.)) != iden");
}

#[test]
pub fn test_msm() {
    let mut rng = TestRng::default();
    let test_size = 10;
    let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, test_size);

    // let t = bases.to_bytes_le();

    // let cpu = snarkvm::cpu::msm(bases.as_slice(), scalars.as_slice());
    let cuda = snarkvm::cuda::msm_cuda(bases.as_slice(), scalars.as_slice()).unwrap();
    // let inner_cuda = cuda::msm_cuda(bases.as_slice(), scalars.as_slice()).unwrap();

    // assert_eq!(cpu, inner_cuda);
    // assert_eq!(cuda, inner_cuda);

    let opencl = opencl::msm_opencl(bases.as_slice(), scalars.as_slice()).unwrap();

    assert_eq!(cuda, opencl);
    // assert_eq!(inner_cuda, opencl);
    // assert_eq!(cpu, opencl);
}

#[test]
fn u64_from_hex_str() {
    let hex_string = "0x1ae3a4617c510ea";
    let hex_string = hex_string.strip_prefix("0x").unwrap_or(hex_string);
    let num = u64::from_str_radix(hex_string, 16).expect("invalid hex string");
    println!("{}", num);
}
