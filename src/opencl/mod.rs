use std::any::TypeId;

use rust_gpu_tools::GPUError;
use snarkvm::cuda::CudaRequest;
use snarkvm::{
    circuit::PrimeField, prelude::AffineCurve, BitIteratorBE, Fr, G1Affine, G1Projective,
};
use snarkvm::{initialize_cuda_request_handler, prelude::*};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

lazy_static::lazy_static! {
    static ref CUDA_DISPATCH: crossbeam_channel::Sender<CudaRequest> = {
        let (sender, receiver) = crossbeam_channel::bounded(4096);
        std::thread::spawn(move || initialize_cuda_request_handler(receiver));
        sender
    };
}

const SCALAR_BITS: usize = 253;
const BIT_WIDTH: usize = 1;
const LIMB_COUNT: usize = 6;
const WINDOW_SIZE: u32 = 128; // must match in cuda source

// pub struct OpenclRequest {
//     bases: Vec<G1Affine>,
//     scalars: Vec<Fr>,
//     response: crossbeam_channel::Sender<Result<G1Projective, GPUError>>,
// }
/// Initialize the cuda request handler.
// fn initialize_opencl_request_handler(input: crossbeam_channel::Receiver<CudaRequest>) {
//     match load_cuda_program() {
//         Ok(program) => {
//             let num_groups = (SCALAR_BITS + BIT_WIDTH - 1) / BIT_WIDTH;

//             let mut context = CudaContext {
//                 num_groups: num_groups as u32,
//                 pixel_func_name: "msm6_pixel".to_string(),
//                 row_func_name: "msm6_collapse_rows".to_string(),
//                 program,
//             };

//             // Handle each cuda request received from the channel.
//             while let Ok(request) = input.recv() {
//                 let out = handle_cuda_request(&mut context, &request);

//                 request.response.send(out).ok();
//             }
//         }
//         Err(err) => {
//             eprintln!("Error loading cuda program: {:?}", err);
//             // If the cuda program fails to load, notify the cuda request dispatcher.
//             while let Ok(request) = input.recv() {
//                 request.response.send(Err(GPUError::DeviceNotFound)).ok();
//             }
//         }
//     }
// }

#[allow(clippy::transmute_undefined_repr)]
pub fn msm_opencl<G: AffineCurve>(
    mut bases: &[G],
    mut scalars: &[<G::ScalarField as PrimeField>::BigInteger],
) -> Result<G::Projective, GPUError> {
    if TypeId::of::<G>() != TypeId::of::<G1Affine>() {
        unimplemented!("trying to use cuda for unsupported curve");
    }

    match bases.len() < scalars.len() {
        true => scalars = &scalars[..bases.len()],
        false => bases = &bases[..scalars.len()],
    }

    if scalars.len() < 4 {
        let mut acc = G::Projective::zero();

        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            acc += &base.mul_bits(BitIteratorBE::new(*scalar))
        }
        return Ok(acc);
    }

    let (sender, receiver) = crossbeam_channel::bounded(1);
    CUDA_DISPATCH
        .send(CudaRequest {
            bases: unsafe { std::mem::transmute(bases.to_vec()) },
            scalars: unsafe { std::mem::transmute(scalars.to_vec()) },
            response: sender,
        })
        .map_err(|_| GPUError::DeviceNotFound)?;
    match receiver.recv() {
        Ok(x) => unsafe { std::mem::transmute_copy(&x) },
        Err(_) => Err(GPUError::DeviceNotFound),
    }
}
