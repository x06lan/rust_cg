use std::sync::Arc;

use compute_shader::compute_shader;
use copy_buffer::copy_buffer;
use vulkano::{
    device::{physical::PhysicalDevice, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    VulkanLibrary,
};

mod compute_shader;
mod copy_buffer;

#[allow(unused)]
fn main() -> anyhow::Result<()> {
    let library = VulkanLibrary::new()?;
    let instance = Instance::new(library, InstanceCreateInfo::default())?;

    let physical_device = instance
        .enumerate_physical_devices()?
        .next()
        .expect("no devices available") as Arc<PhysicalDevice>;

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, properties)| properties.queue_flags.contains(QueueFlags::GRAPHICS))
        .unwrap() as u32;

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )?;

    let queue = queues.next().unwrap();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    copy_buffer(device.clone(), queue.clone(), memory_allocator.clone())?;

    compute_shader(device.clone(), queue.clone(), memory_allocator.clone())?;

    Ok(())
}
