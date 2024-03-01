use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

mod shader;

pub fn compute_shader(
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
) -> anyhow::Result<()> {
    let data_iter = 0..65535u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )?;

    let shader = shader::load(device.clone())?;
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())?,
    )?;

    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )?;

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let work_group_counts = [1024, 1, 1];

    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            descriptor_set,
        )?
        .dispatch(work_group_counts)?;

    let command_buffer = command_buffer_builder.build()?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?;

    future.wait(None)?;

    let content = data_buffer.read()?;
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    if false {
        for v in content.iter() {
            println!("{}", *v);
        }
    }

    println!("Everything succeeded!");

    Ok(())
}
