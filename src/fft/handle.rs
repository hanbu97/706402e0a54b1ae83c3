use std::sync::Arc;

use parking_lot::RwLock;
// use super::{OpenclContext, OpenclRequest, BIT_WIDTH, LIMB_COUNT, WINDOW_SIZE};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use rust_gpu_tools::{program_closures, GPUError, LocalBuffer, Program};
use snarkvm::{
    prelude::{AffineCurve, ProjectiveCurve},
    Fq, Fr, G1Affine, G1Projective, PrimeField, Zero,
};

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
struct OpenclAffine {
    x: Fq,
    y: Fq,
}

// (a: &mut [Fr], omega: Fr, log_n: u32) {
pub struct OpenclRequest {
    pub input: Arc<RwLock<Vec<Fr>>>,
    pub omega: Fr,
    pub log_n: u32,
    pub response: crossbeam_channel::Sender<Result<(), GPUError>>,
}

const MAX_LOG2_RADIX: u32 = 8; // Radix256
use snarkvm::prelude::*;
const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

pub struct OpenclContext {
    pub fft_func_name: String,
    pub program: Program,
}

// Run the OPENCL MSM operation for a given request.
pub fn handle_opencl_request(
    context: &mut OpenclContext,
    request: &mut OpenclRequest,
) -> Result<(), GPUError> {
    // let mut input = request.input;
    let input = &mut request.input.write();
    let log_n = request.log_n;
    let omega = &request.omega;

    let closures = program_closures!(|program, input: &mut [Fr]| -> Result<(), GPUError> {
        let n = 1 << log_n;

        let mut src_buffer = unsafe { program.create_buffer::<Fr>(n)? };
        let mut dst_buffer = unsafe { program.create_buffer::<Fr>(n)? };

        let max_deg = std::cmp::min(MAX_LOG2_RADIX, log_n);

        let mut pq = vec![Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] = Fr::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }
        let pq_buffer = program.create_buffer_from_slice(&pq)?;

        let mut omegas = vec![Fr::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

        program.write_from_buffer(&mut src_buffer, &*input)?;

        let mut log_p = 0u32;
        while log_p < log_n {
            // 1=>radix2, 2=>radix4, 3=>radix8, ...
            let deg = std::cmp::min(max_deg, log_n - log_p);

            let n = 1u32 << log_n;
            let local_work_size = 1 << std::cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
            let global_work_size = n >> deg;
            // let kernel_name = format!("FIELD_radix_fft");
            let kernel = program.create_kernel(
                &context.fft_func_name,
                global_work_size as usize,
                local_work_size as usize,
            )?;
            kernel
                .arg(&src_buffer)
                .arg(&dst_buffer)
                .arg(&pq_buffer)
                .arg(&omegas_buffer)
                .arg(&LocalBuffer::<Fr>::new(1 << deg))
                .arg(&n)
                .arg(&log_p)
                .arg(&deg)
                .arg(&max_deg)
                .run()?;

            log_p += deg;
            std::mem::swap(&mut src_buffer, &mut dst_buffer);
        }

        program.read_into_buffer(&src_buffer, input)?;
        return Ok(());
    });

    context.program.run(closures, input)?;

    Ok(())
}
