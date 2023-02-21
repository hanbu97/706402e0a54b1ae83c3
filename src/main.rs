use snarkvm::{create_scalar_bases, prelude::TestRng, Fr, G1Affine};

pub use snarkvm;
pub mod opencl;

pub fn main() {
    let mut rng = TestRng::default();
    let test_size = 1000;
    let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, test_size);

    let cpu = snarkvm::cpu::msm(bases.as_slice(), scalars.as_slice());
    let cuda = snarkvm::cuda::msm_cuda(bases.as_slice(), scalars.as_slice()).unwrap();

    let opencl = opencl::msm_opencl(bases.as_slice(), scalars.as_slice()).unwrap();

    // assert_eq!(cpu, cuda);
    assert_eq!(cpu, opencl);
}


#[test]
fn u64_from_hex_str() {
    let hex_string = "0x1ae3a4617c510ea";
    let hex_string = hex_string.strip_prefix("0x").unwrap_or(hex_string);
    let num = u64::from_str_radix(hex_string, 16).expect("invalid hex string");
    println!("{}", num);
}
