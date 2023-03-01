use snarkvm::{
    algorithms::fft::EvaluationDomain,
    create_scalar_bases,
    prelude::{TestRng, ToBytes, Uniform},
    Fr, G1Affine, G1Projective, Zero,
};

pub use snarkvm;
pub mod cuda;
pub mod opencl;

fn main() {}

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
pub fn test_fft_fr() {
    let rng = &mut TestRng::default();
    let log_d = 9;
    let d = 1 << log_d;
    let domain = EvaluationDomain::<Fr>::new(d).unwrap();

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
