use std::sync::Arc;

use parking_lot::RwLock;
use rust_gpu_tools::GPUError;
use snarkvm::{BitIteratorBE, Fr, G1Affine, G1Projective, PrimeField, Zero};

use self::{
    handle::{handle_opencl_request, OpenclContext, OpenclRequest},
    program::load_opencl_program,
};
pub mod handle;
pub mod program;

// const SCALAR_BITS: usize = 253;
// const BIT_WIDTH: usize = 1;
// const LIMB_COUNT: usize = 6;
// const WINDOW_SIZE: u32 = 128; // must match in opencl source

lazy_static::lazy_static! {
    static ref OPENCL_DISPATCH: crossbeam_channel::Sender<OpenclRequest> = {
        let (sender, receiver) = crossbeam_channel::bounded(4096);
        std::thread::spawn(move || initialize_opencl_request_handler(receiver));
        sender
    };
}

/// Initialize the opencl request handler.
fn initialize_opencl_request_handler(input: crossbeam_channel::Receiver<OpenclRequest>) {
    match load_opencl_program() {
        Ok(program) => {
            // let num_groups = (SCALAR_BITS + BIT_WIDTH - 1) / BIT_WIDTH;

            let mut context = OpenclContext {
                fft_func_name: "FIELD_radix_fft".to_string(),
                program,
            };

            // Handle each opencl request received from the channel.
            while let Ok(mut request) = input.recv() {
                handle_opencl_request(&mut context, &mut request).ok();
                request.response.send(Ok(())).ok();
            }
        }
        Err(err) => {
            eprintln!("Error loading opencl program: {:?}", err);
            // If the opencl program fails to load, notify the opencl request dispatcher.
            while let Ok(request) = input.recv() {
                request.response.send(Err(GPUError::DeviceNotFound)).ok();
            }
        }
    }
}

#[allow(clippy::transmute_undefined_repr)]
pub fn fft_opencl(input: Arc<RwLock<Vec<Fr>>>, omega: Fr, log_n: u32) -> Result<(), GPUError> {
    // pub fn fft_opencl(input: &mut [], omega: Fr, log_n: u32) -> Result<(), GPUError> {
    let (sender, receiver) = crossbeam_channel::bounded(1);
    OPENCL_DISPATCH
        .send(OpenclRequest {
            input,
            log_n,
            omega,
            response: sender,
        })
        .map_err(|_| GPUError::DeviceNotFound)?;

    if let Err(_) = receiver.recv() {
        Err(GPUError::DeviceNotFound)
    } else {
        Ok(())
    }
}
