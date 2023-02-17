use snarkvm::{create_scalar_bases, prelude::TestRng, Fr, G1Affine};

pub use snarkvm;
pub mod opencl;

pub fn main() {
    let mut rng = TestRng::default();
    let test_size = 1000;
    let (bases, scalars) = create_scalar_bases::<G1Affine, Fr>(&mut rng, test_size);

    let cpu = snarkvm::cpu::msm(bases.as_slice(), scalars.as_slice());
    // let cuda = snarkvm::cuda::msm_cuda(bases.as_slice(), scalars.as_slice()).unwrap();

    let opencl = opencl::msm_opencl(bases.as_slice(), scalars.as_slice()).unwrap();

    assert_eq!(cpu, opencl);
}
