// use snarkvm_utilities::BitIteratorBE;

use rust_gpu_tools::{Device, GPUError, Program};
use snarkvm::{prelude::AffineCurve, G1Affine, PrimeField};
use snarkvm::{BitIteratorBE, Fr, G1Projective, Zero};
use std::path::{Path, PathBuf};
// use std::{any::TypeId, path::PathBuf};
use std::{any::TypeId, process::Command};
pub mod handle;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use self::handle::handle_cuda_request;

const SCALAR_BITS: usize = 253;
const BIT_WIDTH: usize = 1;
const LIMB_COUNT: usize = 6;
const WINDOW_SIZE: u32 = 128; // must match in cuda source

const ALEO_DIRECTORY: &str = ".aleo";

pub struct CudaRequest {
    pub bases: Vec<G1Affine>,
    pub scalars: Vec<Fr>,
    pub response: crossbeam_channel::Sender<Result<G1Projective, GPUError>>,
}

pub fn aleo_dir() -> PathBuf {
    // Locate the home directory as the starting point.
    // If called on a non-standard OS, use the repository directory.
    let mut path = match dirs::home_dir() {
        Some(home) => home,
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")),
    };
    // Append the Aleo directory to the path.
    path.push(ALEO_DIRECTORY);
    path
}

/// Generates the cuda msm binary.
fn generate_cuda_binary<P: AsRef<Path>>(file_path: P, debug: bool) -> Result<(), GPUError> {
    // Find the latest compute code values.
    let nvcc_help = Command::new("nvcc").arg("-h").output()?.stdout;
    let nvcc_output = std::str::from_utf8(&nvcc_help)
        .map_err(|_| GPUError::Generic("Missing nvcc command".to_string()))?;

    // Generate the parent directory.
    let mut resource_path = aleo_dir();
    resource_path.push("resources/cuda/");
    std::fs::create_dir_all(resource_path)?;

    // TODO (raychu86): Fix this approach to generating files. Should just read all files in the `blst_377_cuda` directory.
    // Store the `.cu` and `.h` files temporarily for fatbin generation
    let mut asm_cuda_path = aleo_dir();
    let mut asm_cuda_h_path = aleo_dir();
    asm_cuda_path.push("resources/cuda/asm_cuda.cu");
    asm_cuda_h_path.push("resources/cuda/asm_cuda.h");

    let mut blst_377_ops_path = aleo_dir();
    let mut blst_377_ops_h_path = aleo_dir();
    blst_377_ops_path.push("resources/cuda/blst_377_ops.cu");
    blst_377_ops_h_path.push("resources/cuda/blst_377_ops.h");

    let mut msm_path = aleo_dir();
    msm_path.push("resources/cuda/msm.cu");

    let mut types_path = aleo_dir();
    types_path.push("resources/cuda/types.h");

    let mut tests_path = aleo_dir();
    tests_path.push("resources/cuda/tests.cu");

    // Write all the files to the relative path.
    {
        let asm_cuda = include_bytes!("./blst_377_cuda/asm_cuda.cu");
        let asm_cuda_h = include_bytes!("./blst_377_cuda/asm_cuda.h");
        std::fs::write(&asm_cuda_path, asm_cuda)?;
        std::fs::write(&asm_cuda_h_path, asm_cuda_h)?;

        let blst_377_ops = include_bytes!("./blst_377_cuda/blst_377_ops.cu");
        let blst_377_ops_h = include_bytes!("./blst_377_cuda/blst_377_ops.h");
        std::fs::write(&blst_377_ops_path, blst_377_ops)?;
        std::fs::write(&blst_377_ops_h_path, blst_377_ops_h)?;

        let msm = include_bytes!("./blst_377_cuda/msm.cu");
        std::fs::write(&msm_path, msm)?;

        let types = include_bytes!("./blst_377_cuda/types.h");
        std::fs::write(&types_path, types)?;
    }

    // Generate the cuda fatbin.
    let mut command = Command::new("nvcc");
    command
        .arg(asm_cuda_path.as_os_str())
        .arg(blst_377_ops_path.as_os_str())
        .arg(msm_path.as_os_str());

    // Add the debug feature for tests.
    if debug {
        let tests = include_bytes!("./blst_377_cuda/tests.cu");
        std::fs::write(&tests_path, tests)?;

        command.arg(tests_path.as_os_str()).arg("--device-debug");
    }

    // Add supported gencodes
    command
        .arg("--generate-code=arch=compute_60,code=sm_60")
        .arg("--generate-code=arch=compute_70,code=sm_70")
        .arg("--generate-code=arch=compute_75,code=sm_75");

    if nvcc_output.contains("compute_80") {
        command.arg("--generate-code=arch=compute_80,code=sm_80");
    }

    if nvcc_output.contains("compute_86") {
        command.arg("--generate-code=arch=compute_86,code=sm_86");
    }

    command
        .arg("-fatbin")
        .arg("-dlink")
        .arg("-o")
        .arg(file_path.as_ref().as_os_str());

    eprintln!("\nRunning command: {:?}", command);

    let status = command.status()?;

    // Delete all the temporary .cu and .h files.
    {
        let _ = std::fs::remove_file(asm_cuda_path);
        let _ = std::fs::remove_file(asm_cuda_h_path);
        let _ = std::fs::remove_file(blst_377_ops_path);
        let _ = std::fs::remove_file(blst_377_ops_h_path);
        let _ = std::fs::remove_file(msm_path);
        let _ = std::fs::remove_file(types_path);
        let _ = std::fs::remove_file(tests_path);
    }

    // Execute the command.
    if !status.success() {
        return Err(GPUError::KernelNotFound(
            "Could not generate a new msm kernel".to_string(),
        ));
    }

    Ok(())
}

/// Generates the cuda msm binary.
// fn generate_cuda_binary<P: AsRef<Path>>(file_path: P, debug: bool) -> Result<(), GPUError> {
//     // Find the latest compute code values.
//     let nvcc_help = Command::new("nvcc").arg("-h").output()?.stdout;
//     let nvcc_output = std::str::from_utf8(&nvcc_help)
//         .map_err(|_| GPUError::Generic("Missing nvcc command".to_string()))?;

//     // Generate the parent directory.
//     // let mut resource_path = aleo_dir();
//     // resource_path.push("resources/cuda/");
//     let resource_path = std::path::PathBuf::from("./cuda/");
//     std::fs::create_dir_all(resource_path)?;

//     // TODO (raychu86): Fix this approach to generating files. Should just read all files in the `cuda` directory.
//     // Store the `.cu` and `.h` files temporarily for fatbin generation
//     // let mut asm_cuda_path = aleo_dir();
//     // let mut asm_cuda_h_path = aleo_dir();
//     // asm_cuda_path.push("resources/cuda/asm_cuda.cu");
//     // asm_cuda_h_path.push("resources/cuda/asm_cuda.h");
//     let asm_cuda_path = std::path::PathBuf::from("./cuda/asm_cuda.cu");
//     let asm_cuda_h_path = std::path::PathBuf::from("./cuda/asm_cuda.h");

//     dbg!(&asm_cuda_h_path);

//     // let mut blst_377_ops_path = aleo_dir();
//     // let mut blst_377_ops_h_path = aleo_dir();
//     // blst_377_ops_path.push("resources/cuda/blst_377_ops.cu");
//     // blst_377_ops_h_path.push("resources/cuda/blst_377_ops.h");
//     let blst_377_ops_path = std::path::PathBuf::from("./cuda/blst_377_ops.cu");
//     let blst_377_ops_h_path = std::path::PathBuf::from("./cuda/blst_377_ops.cu");

//     // let mut msm_path = aleo_dir();
//     // msm_path.push("resources/cuda/msm.cu");
//     let msm_path = std::path::PathBuf::from("./cuda/msm.cu");

//     // let mut types_path = aleo_dir();
//     // types_path.push("resources/cuda/types.h");
//     let types_path = std::path::PathBuf::from("./cuda/types.cu");

//     // let mut tests_path = aleo_dir();
//     // tests_path.push("resources/cuda/tests.cu");
//     let tests_path = std::path::PathBuf::from("./cuda/tests.cu");

//     // Write all the files to the relative path.
//     {
//         let asm_cuda = include_bytes!("./blst_377_cuda/asm_cuda.cu");
//         let asm_cuda_h = include_bytes!("./blst_377_cuda/asm_cuda.h");
//         std::fs::write(&asm_cuda_path, asm_cuda)?;
//         std::fs::write(&asm_cuda_h_path, asm_cuda_h)?;

//         let blst_377_ops = include_bytes!("./blst_377_cuda/blst_377_ops.cu");
//         let blst_377_ops_h = include_bytes!("./blst_377_cuda/blst_377_ops.h");
//         std::fs::write(&blst_377_ops_path, blst_377_ops)?;
//         std::fs::write(&blst_377_ops_h_path, blst_377_ops_h)?;

//         let msm = include_bytes!("./blst_377_cuda/msm.cu");
//         std::fs::write(&msm_path, msm)?;

//         let types = include_bytes!("./blst_377_cuda/types.h");
//         std::fs::write(&types_path, types)?;
//     }

//     // Generate the cuda fatbin.
//     let mut command = Command::new("nvcc");
//     command
//         .arg(asm_cuda_path.as_os_str())
//         .arg(blst_377_ops_path.as_os_str())
//         .arg(msm_path.as_os_str());

//     // Add the debug feature for tests.
//     if debug {
//         let tests = include_bytes!("./blst_377_cuda/tests.cu");
//         std::fs::write(&tests_path, tests)?;

//         command.arg(tests_path.as_os_str()).arg("--device-debug");
//     }

//     // Add supported gencodes
//     command
//         .arg("--generate-code=arch=compute_60,code=sm_60")
//         .arg("--generate-code=arch=compute_70,code=sm_70")
//         .arg("--generate-code=arch=compute_75,code=sm_75");

//     if nvcc_output.contains("compute_80") {
//         command.arg("--generate-code=arch=compute_80,code=sm_80");
//     }

//     if nvcc_output.contains("compute_86") {
//         command.arg("--generate-code=arch=compute_86,code=sm_86");
//     }

//     command
//         .arg("-fatbin")
//         .arg("-dlink")
//         .arg("-o")
//         .arg(file_path.as_ref().as_os_str());

//     eprintln!("\nRunning command: {:?}", command);

//     let status = command.status()?;

//     // Delete all the temporary .cu and .h files.
//     {
//         let _ = std::fs::remove_file(asm_cuda_path);
//         let _ = std::fs::remove_file(asm_cuda_h_path);
//         let _ = std::fs::remove_file(blst_377_ops_path);
//         let _ = std::fs::remove_file(blst_377_ops_h_path);
//         let _ = std::fs::remove_file(msm_path);
//         let _ = std::fs::remove_file(types_path);
//         let _ = std::fs::remove_file(tests_path);
//     }

//     // Execute the command.
//     if !status.success() {
//         return Err(GPUError::KernelNotFound(
//             "Could not generate a new msm kernel".to_string(),
//         ));
//     }

//     Ok(())
// }

/// Loads the msm.fatbin into an executable CUDA program.
fn load_cuda_program() -> Result<Program, GPUError> {
    let devices: Vec<_> = Device::all();
    let device = match devices.first() {
        Some(device) => device,
        None => return Err(GPUError::DeviceNotFound),
    };

    // Find the path to the msm fatbin kernel
    // let mut file_path = aleo_dir();
    // file_path.push("resources/cuda/msm.fatbin");

    let mut file_path = std::path::PathBuf::from("./msm.fatbin");

    // If the file does not exist, regenerate the fatbin.
    if !file_path.exists() {
        generate_cuda_binary(&file_path, false)?;
    }

    let cuda_device = match device.cuda_device() {
        Some(device) => device,
        None => return Err(GPUError::DeviceNotFound),
    };

    eprintln!(
        "\nUsing '{}' as CUDA device with {} bytes of memory",
        device.name(),
        device.memory()
    );

    let cuda_kernel = std::fs::read(file_path.clone())?;

    // Load the cuda program from the kernel bytes.
    let cuda_program = match rust_gpu_tools::cuda::Program::from_bytes(cuda_device, &cuda_kernel) {
        Ok(program) => program,
        Err(err) => {
            // Delete the failing cuda kernel.
            std::fs::remove_file(file_path)?;
            return Err(err);
        }
    };

    Ok(Program::Cuda(cuda_program))
}

pub struct CudaContext {
    num_groups: u32,
    pixel_func_name: String,
    row_func_name: String,
    program: Program,
}

/// Initialize the cuda request handler.
pub fn initialize_cuda_request_handler(input: crossbeam_channel::Receiver<CudaRequest>) {
    match load_cuda_program() {
        Ok(program) => {
            let num_groups = (SCALAR_BITS + BIT_WIDTH - 1) / BIT_WIDTH;

            let mut context = CudaContext {
                num_groups: num_groups as u32,
                pixel_func_name: "msm6_pixel".to_string(),
                row_func_name: "msm6_collapse_rows".to_string(),
                program,
            };

            // Handle each cuda request received from the channel.
            while let Ok(request) = input.recv() {
                let out = handle_cuda_request(&mut context, &request);

                request.response.send(out).ok();
            }
        }
        Err(err) => {
            eprintln!("Error loading cuda program: {:?}", err);
            // If the cuda program fails to load, notify the cuda request dispatcher.
            while let Ok(request) = input.recv() {
                request.response.send(Err(GPUError::DeviceNotFound)).ok();
            }
        }
    }
}

lazy_static::lazy_static! {
    static ref CUDA_DISPATCH: crossbeam_channel::Sender<CudaRequest> = {
        let (sender, receiver) = crossbeam_channel::bounded(4096);
        std::thread::spawn(move || initialize_cuda_request_handler(receiver));
        sender
    };
}

#[allow(clippy::transmute_undefined_repr)]
pub fn msm_cuda<G: AffineCurve>(
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
