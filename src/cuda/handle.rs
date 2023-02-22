use super::BIT_WIDTH;
use super::{CudaContext, CudaRequest, LIMB_COUNT, WINDOW_SIZE};
use rust_gpu_tools::program_closures;
use rust_gpu_tools::GPUError;
use snarkvm::prelude::ProjectiveCurve;
use snarkvm::Zero;
use snarkvm::{prelude::AffineCurve, Fq, G1Affine, G1Projective, PrimeField};

/// Creates parallel iterator over refs if `parallel` feature is enabled.
#[macro_export]
macro_rules! cfg_iter {
    ($e: expr) => {{
        #[cfg(feature = "parallel")]
        let result = $e.par_iter();

        #[cfg(not(feature = "parallel"))]
        let result = $e.iter();

        result
    }};
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
#[repr(C)]
struct CudaAffine {
    x: Fq,
    y: Fq,
}

/// Run the CUDA MSM operation for a given request.
pub fn handle_cuda_request(
    context: &mut CudaContext,
    request: &CudaRequest,
) -> Result<G1Projective, GPUError> {
    let mapped_bases: Vec<_> = cfg_iter!(request.bases)
        .map(|affine| CudaAffine {
            x: affine.x,
            y: affine.y,
        })
        .collect();

    let mut window_lengths = (0..(request.scalars.len() as u32 / WINDOW_SIZE))
        .into_iter()
        .map(|_| WINDOW_SIZE)
        .collect::<Vec<u32>>();
    let overflow_size = request.scalars.len() as u32 - window_lengths.len() as u32 * WINDOW_SIZE;
    if overflow_size > 0 {
        window_lengths.push(overflow_size);
    }

    let closures = program_closures!(|program, _arg| -> Result<Vec<u8>, GPUError> {
        let window_lengths_buffer = program.create_buffer_from_slice(&window_lengths)?;
        let base_buffer = program.create_buffer_from_slice(&mapped_bases)?;
        let scalars_buffer = program.create_buffer_from_slice(&request.scalars)?;

        let buckets_length = context.num_groups as usize
            * window_lengths.len() as usize
            * 8
            * LIMB_COUNT as usize
            * 3;
        dbg!(buckets_length);

        let buckets_buffer = program.create_buffer_from_slice(&vec![
            0u8;
            context.num_groups as usize
                * window_lengths.len()
                    as usize
                * 8
                * LIMB_COUNT as usize
                * 3
        ])?;
        let result_buffer = program.create_buffer_from_slice(&vec![
            0u8;
            LIMB_COUNT as usize
                * 8
                * context.num_groups
                    as usize
                * 3
        ])?;

        // // The global work size follows CUDA's definition and is the number of
        // // `LOCAL_WORK_SIZE` sized thread groups.
        // const LOCAL_WORK_SIZE: usize = 256;
        // let global_work_size =
        //     (window_lengths.len() * context.num_groups as usize + LOCAL_WORK_SIZE - 1) / LOCAL_WORK_SIZE;

        let kernel_1 = program.create_kernel(
            &context.pixel_func_name,
            window_lengths.len(),
            context.num_groups as usize,
        )?;

        kernel_1
            .arg(&buckets_buffer)
            .arg(&base_buffer)
            .arg(&scalars_buffer)
            .arg(&window_lengths_buffer)
            .arg(&(window_lengths.len() as u32))
            .run()?;

        let mut buckets_results = vec![0u8; buckets_length];
        program.read_into_buffer(&buckets_buffer, &mut buckets_results)?;
        use std::io::Write;
        let mut file = std::fs::File::create("../cuda.txt").unwrap();
        writeln!(file, "{:?}", buckets_results).expect("cuda txt write error");

        let kernel_2 =
            program.create_kernel(&context.row_func_name, 1, context.num_groups as usize)?;

        kernel_2
            .arg(&result_buffer)
            .arg(&buckets_buffer)
            .arg(&(window_lengths.len() as u32))
            .run()?;

        let mut results = vec![0u8; LIMB_COUNT as usize * 8 * context.num_groups as usize * 3];
        program.read_into_buffer(&result_buffer, &mut results)?;

        Ok(results)
    });

    let mut out = context.program.run(closures, ())?;

    let base_size =
        std::mem::size_of::<<<G1Affine as AffineCurve>::BaseField as PrimeField>::BigInteger>();

    let windows = unsafe {
        Vec::from_raw_parts(
            out.as_mut_ptr() as *mut G1Projective,
            out.len() / base_size / 3,
            out.capacity() / base_size / 3,
        )
    };
    std::mem::forget(out);

    let lowest = windows.first().unwrap();

    // We're traversing windows from high to low.
    let final_result = windows[1..]
        .iter()
        .rev()
        .fold(G1Projective::zero(), |mut total, sum_i| {
            total += sum_i;
            for _ in 0..BIT_WIDTH {
                total.double_in_place();
            }
            total
        })
        + lowest;
    Ok(final_result)
}
