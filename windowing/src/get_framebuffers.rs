use std::sync::Arc;

use vulkano::{
    image::{view::ImageView, Image},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

pub fn get_framebuffers(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> anyhow::Result<Vec<Arc<Framebuffer>>> {
    let result = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect();

    Ok(result)
}
