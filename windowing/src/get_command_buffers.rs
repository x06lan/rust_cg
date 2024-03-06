use std::sync::Arc;

use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
    },
    device::Queue,
    pipeline::GraphicsPipeline,
    render_pass::Framebuffer,
};

use crate::shaders::Vert;

pub fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &[Arc<Framebuffer>],
    vertex_buffer: &Subbuffer<[Vert]>,
) -> anyhow::Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
    let result = framebuffers
        .iter()
        .map(|framebuffer| {
            get_framebuffer(
                framebuffer,
                queue,
                command_buffer_allocator,
                pipeline,
                vertex_buffer,
            )
            .unwrap()
        })
        .collect();

    Ok(result)
}

fn get_framebuffer(
    framebuffer: &Arc<Framebuffer>,
    queue: &Arc<Queue>,
    command_buffer_allocator: &StandardCommandBufferAllocator,
    pipeline: &Arc<GraphicsPipeline>,
    vertex_buffer: &Subbuffer<[Vert]>,
) -> anyhow::Result<Arc<PrimaryAutoCommandBuffer>> {
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    builder
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?
        .bind_pipeline_graphics(pipeline.clone())?
        .bind_vertex_buffers(0, vertex_buffer.clone())?
        .draw(vertex_buffer.len() as u32, 1, 0, 0)?
        .end_render_pass(SubpassEndInfo::default())?;

    let result = builder.build()?;

    Ok(result)
}
