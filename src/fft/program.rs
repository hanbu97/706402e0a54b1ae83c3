use rust_gpu_tools::{Device, GPUError, Program};

/// Loads the msm.fatbin into an executable CUDA program.
pub fn load_opencl_program() -> Result<Program, GPUError> {
    let devices: Vec<_> = Device::all();
    let device = match devices.first() {
        Some(device) => device,
        None => return Err(GPUError::DeviceNotFound),
    };

    let opencl_device = match device.opencl_device() {
        Some(device) => device,
        None => return Err(GPUError::DeviceNotFound),
    };

    eprintln!(
        "\nUsing '{}' as OPENCL device with {} bytes of memory",
        device.name(),
        device.memory()
    );

    // let opencl_kernel = std::fs::read(file_path.clone())?;
    let opencl_kernel = include_str!("./fft.cl");

    // Load the cuda program from the kernel bytes.
    let cuda_program =
        match rust_gpu_tools::opencl::Program::from_opencl(opencl_device, opencl_kernel) {
            Ok(program) => program,
            Err(err) => {
                // Delete the failing cuda kernel.
                // std::fs::remove_file(file_path)?;
                return Err(err);
            }
        };

    Ok(Program::Opencl(cuda_program))
}

#[test]
fn test_load_opencl_program() {
    if let Err(e) = load_opencl_program() {
        dbg!(e);
    };
}
