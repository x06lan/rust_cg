use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    sync::{self, GpuFuture},
};

pub fn copy_buffer(
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> anyhow::Result<()> {
    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    )?;

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_content,
    )?;

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))?;

    let command_buffer = builder.build()?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    future.wait(None)?;

    let src_content = source.read()?;
    let destination_content = destination.read()?;
    assert_eq!(&*src_content, &*destination_content);

    if false {
        for c in destination_content.iter() {
            println!("{}", *c);
        }
    }

    println!("Everything succeeded!");

    Ok(())
}
