use std::sync::Arc;

use vulkano::{device::Device, render_pass::RenderPass, swapchain::Swapchain};

pub fn get_render_pass(
    device: Arc<Device>,
    swapchain: &Arc<Swapchain>,
) -> anyhow::Result<Arc<RenderPass>> {
    let result = vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {},
        }
    )?;

    Ok(result)
}
