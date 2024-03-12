// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#![allow(clippy::eq_op)]

use std::{sync::Arc, time::Instant};

use cgmath::{Matrix3, Matrix4, Point3, Rad, SquareMatrix, Transform, Transform3, Vector3};
use egui::{epaint::Shadow, style::Margin, vec2, Align, Align2, Color32, Frame, Rounding, Window};
use egui_winit_vulkano::egui::ScrollArea;
use egui_winit_vulkano::{egui, Gui, GuiConfig};
use vulkano::descriptor_set::layout::{DescriptorSetLayoutCreateFlags, DescriptorType};
use vulkano::pipeline::{DynamicState, Pipeline, PipelineBindPoint};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::{ClearValue, Format},
    image::{view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage, SampleCount},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::platform::run_return;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};
// Render a triangle (scene) and a gui from a subpass on top of it (with some transparent fill)

pub fn main() {
    // Winit event loop
    let event_loop = EventLoop::new();
    // Vulkano context
    let context = VulkanoContext::new(VulkanoConfig::default());
    // Vulkano windows (create one)
    let mut windows = VulkanoWindows::default();
    windows.create_window(&event_loop, &context, &WindowDescriptor::default(), |ci| {
        ci.image_format = vulkano::format::Format::B8G8R8A8_UNORM;
        ci.min_image_count = ci.min_image_count.max(2);
    });
    // Create out gui pipeline
    let extent: [u32; 2] = windows
        .get_primary_renderer_mut()
        .unwrap()
        .swapchain_image_size();
    let mut values = Values {
        tx: 0.0,
        ty: 0.0,
        tz: 0.0,
        rx: 180.0,
        ry: 180.0,
        rz: 180.0,
    };
    let mut gui_pipeline = SimpleGuiPipeline::new(
        context.graphics_queue().clone(),
        windows
            .get_primary_renderer_mut()
            .unwrap()
            .swapchain_format(),
        context.memory_allocator(),
        extent,
    );
    // Create gui subpass
    let mut gui = Gui::new_with_subpass(
        &event_loop,
        windows.get_primary_renderer_mut().unwrap().surface(),
        windows.get_primary_renderer_mut().unwrap().graphics_queue(),
        gui_pipeline.gui_pass(),
        windows
            .get_primary_renderer_mut()
            .unwrap()
            .swapchain_format(),
        GuiConfig::default(),
    );

    let mut check = true;
    // Create gui state (pass anything your state requires)
    event_loop.run(move |event, _, control_flow| {
        let renderer = windows.get_primary_renderer_mut().unwrap();
        match event {
            Event::WindowEvent { event, window_id } if window_id == renderer.window().id() => {
                // Update Egui integration so the UI works!
                let _pass_events_to_game = !gui.update(&event);
                match event {
                    WindowEvent::Resized(_) => {
                        renderer.resize();
                    }
                    WindowEvent::ScaleFactorChanged { .. } => {
                        renderer.resize();
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                    }
                    _ => (),
                }
            }
            Event::RedrawRequested(window_id) if window_id == window_id => {
                // Set immediate UI in redraw here
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();

                    Window::new("Transparent Window").show(&ctx, |ui| {
                        ui.vertical_centered(|ui| {
                            ui.add(egui::widgets::Label::new("Hi there!"));
                            ui.add(egui::widgets::Checkbox::new(&mut check, "Click me!"));
                            ui.label(egui::RichText::new(check.to_string()).size(32.0));
                            ui.add(egui::Slider::new(&mut values.tx, -1.0..=1.0).text("tx"));
                            ui.add(egui::Slider::new(&mut values.ty, -1.0..=1.0).text("ty"));
                            ui.add(egui::Slider::new(&mut values.tz, -1.0..=1.0).text("tz"));
                            ui.add(egui::Slider::new(&mut values.rx, -90.0..=90.0).text("rx"));
                            ui.add(egui::Slider::new(&mut values.ry, -90.0..=90.0).text("ry"));
                            ui.add(egui::Slider::new(&mut values.rz, -90.0..=90.0).text("rz"));
                        });
                        ui.separator();
                    });
                });

                // Acquire swapchain future
                match renderer.acquire() {
                    Ok(future) => {
                        // Render gui
                        let after_future = gui_pipeline.render(
                            future,
                            renderer.swapchain_image_view(),
                            &mut gui,
                            context.memory_allocator(),
                            values.clone(),
                        );

                        // Present swapchain
                        renderer.present(after_future, true);
                    }
                    Err(vulkano::VulkanError::OutOfDate) => {
                        renderer.resize();
                    }
                    Err(e) => panic!("Failed to acquire swapchain future: {}", e),
                };
            }
            Event::MainEventsCleared => {
                renderer.window().request_redraw();
            }
            _ => (),
        }
    });
}

struct Values {
    tx: f32,
    ty: f32,
    tz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
}
impl Clone for Values {
    fn clone(&self) -> Self {
        Self {
            rx: self.rx,
            ry: self.ry,
            rz: self.rz,
            tx: self.tx,
            ty: self.ty,
            tz: self.tz,
        }
    }
}
struct SimpleGuiPipeline {
    start_time: Instant,
    queue: Arc<Queue>,
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass,
    vertex_buffer: Subbuffer<[MyVertex]>,
    index_buffer: Subbuffer<[u32]>,
    vertex_buffer2: Subbuffer<[MyVertex]>,
    index_buffer2: Subbuffer<[u32]>,
    uniform_buffer: Subbuffer<vs::Data>,
    uniform_buffer2: Subbuffer<vs::Data>,
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,
    descriptor_set_allocator2: StandardDescriptorSetAllocator,
}

impl SimpleGuiPipeline {
    pub fn new(
        queue: Arc<Queue>,
        image_format: vulkano::format::Format,
        allocator: &Arc<StandardMemoryAllocator>,
        extent: [u32; 2],
    ) -> Self {
        let render_pass = Self::create_render_pass(queue.device().clone(), image_format);
        let (pipeline, subpass) =
            Self::create_pipeline(queue.device().clone(), render_pass.clone(), extent);

        let vertex_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [
                MyVertex {
                    position: [-1.0, -1.0, -1.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                }, // Red
                MyVertex {
                    position: [1.0, -1.0, -1.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                }, // Green
                MyVertex {
                    position: [1.0, 1.0, -1.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                }, // Blue
                MyVertex {
                    position: [-1.0, 1.0, -1.0],
                    color: [1.0, 1.0, 0.0, 1.0],
                }, // Yellow
                MyVertex {
                    position: [-1.0, -1.0, 1.0],
                    color: [1.0, 0.0, 1.0, 1.0],
                }, // Magenta
                MyVertex {
                    position: [1.0, -1.0, 1.0],
                    color: [0.0, 1.0, 1.0, 1.0],
                }, // Cyan
                MyVertex {
                    position: [1.0, 1.0, 1.0],
                    color: [0.5, 0.5, 0.5, 1.0],
                }, // Gray
                MyVertex {
                    position: [-1.0, 1.0, 1.0],
                    color: [0.7, 0.7, 0.7, 1.0],
                }, // Light Gray
            ],
        )
        .unwrap();
        let vertex_buffer2 = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            [
                MyVertex {
                    position: [100.0, 0.1, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [100.0, -0.1, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [-100.0, -0.1, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [-100.0, 0.1, 0.0],
                    color: [1.0, 0.0, 0.0, 1.0],
                }, // x
                MyVertex {
                    position: [0.1, 100.0, 0.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [-0.1, 100.0, 0.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [-0.1, -100.0, 0.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                },
                MyVertex {
                    position: [0.1, -100.0, 0.0],
                    color: [0.0, 1.0, 0.0, 1.0],
                }, // y
                MyVertex {
                    position: [0.0, 0.1, 100.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                },
                MyVertex {
                    position: [0.0, -0.1, 100.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                },
                MyVertex {
                    position: [0.0, -0.1, -100.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                },
                MyVertex {
                    position: [0.0, 0.1, -100.0],
                    color: [0.0, 0.0, 1.0, 1.0],
                }, // z
            ],
        )
        .unwrap();
        let index_buffer = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                0, 1, 2, 2, 3, 0, // Front face
                1, 5, 6, 6, 2, 1, // Right face
                4, 5, 6, 6, 7, 4, // Back face
                4, 0, 3, 3, 7, 4, // Left face
                3, 2, 6, 6, 7, 3, // Top face
                0, 1, 5, 5, 4, 0, // Bottom face
            ],
        )
        .unwrap();
        let index_buffer2 = Buffer::from_iter(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            [
                0, 3, 1, //
                3, 1, 2, //
                4, 7, 5, //
                7, 5, 6, //
                8, 11, 9, //
                11, 9, 10, //
            ],
        )
        .unwrap();

        let uniform_buffer = Buffer::new_sized(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();
        let uniform_buffer2 = Buffer::new_sized(
            allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        )
        .unwrap();

        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(queue.device().clone(), Default::default());

        let descriptor_set_allocator2 =
            StandardDescriptorSetAllocator::new(queue.device().clone(), Default::default());
        // Create an allocator for command-buffer data
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            queue.device().clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        );
        let start_time = Instant::now();

        Self {
            start_time,
            queue,
            render_pass,
            pipeline,
            subpass,
            vertex_buffer,
            index_buffer,
            vertex_buffer2,
            index_buffer2,
            uniform_buffer,
            uniform_buffer2,
            command_buffer_allocator,
            descriptor_set_allocator,
            descriptor_set_allocator2,
        }
    }

    fn create_render_pass(device: Arc<Device>, format: Format) -> Arc<RenderPass> {
        vulkano::ordered_passes_renderpass!(
            device,
            attachments: {
                color: {
                    format: format,
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: SampleCount::Sample1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            passes: [
                { color: [color], depth_stencil: {depth_stencil}, input: [] }, // Draw what you want on this pass
                { color: [color], depth_stencil: {}, input: [] } // Gui render pass
            ]
        )
        .unwrap()
    }

    fn gui_pass(&self) -> Subpass {
        Subpass::from(self.render_pass.clone(), 1).unwrap()
    }

    fn create_pipeline(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        extent: [u32; 2],
    ) -> (Arc<GraphicsPipeline>, Subpass) {
        let vs = vs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();
        let fs = fs::load(device.clone())
            .expect("failed to create shader module")
            .entry_point("main")
            .unwrap();

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass, 0).unwrap();
        (
            GraphicsPipeline::new(
                device,
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState {
                        viewports: [Viewport {
                            offset: [0.0, 0.0],
                            extent: [extent[0] as f32, extent[1] as f32],
                            depth_range: 0.0..=1.0,
                        }]
                        .into_iter()
                        .collect(),
                        ..Default::default()
                    }),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.clone().into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap(),
            subpass,
        )
    }

    pub fn render(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image: Arc<ImageView>,
        gui: &mut Gui,
        allocator: &Arc<StandardMemoryAllocator>,
        values: Values,
    ) -> Box<dyn GpuFuture> {
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let extent = image.image().extent();
        let depth_buffer = ImageView::new_default(
            Image::new(
                allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent: extent,
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![image, depth_buffer],
                ..Default::default()
            },
        )
        .unwrap();
        let mut model_transfrom = Matrix4::from_angle_x(Rad(values.rx as f32))
            * Matrix4::from_angle_y(Rad(values.ry as f32))
            * Matrix4::from_angle_z(Rad(values.rz as f32));

        model_transfrom =
            model_transfrom * Matrix4::from_translation([values.tx, values.ty, values.tz].into());

        // note: this teapot was meant for OpenGL where the origin is at the lower left
        //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
        let aspect_ratio = extent[0] as f32 / extent[1] as f32;
        let proj = cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.05, 10.0);
        let view = Matrix4::look_at_rh(
            Point3::new(0.1, 0.1, 0.05),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
        );
        let scale = Matrix4::from_scale(0.01);

        let uniform_data = vs::Data {
            world: Matrix4::from(model_transfrom).into(),
            view: (view * scale).into(),
            proj: proj.into(),
        };

        *self.uniform_buffer.write().unwrap() = uniform_data;

        let layout = self.pipeline.layout().set_layouts()[0].clone();
        let set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.uniform_buffer.clone())],
            [],
        )
        .unwrap();
        let uniform_data2 = vs::Data {
            world: Matrix4::identity().into(),
            view: (view * scale).into(),
            proj: proj.into(),
        };

        *self.uniform_buffer2.write().unwrap() = uniform_data2;
        let set2 = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator2,
            layout.clone(),
            [WriteDescriptorSet::buffer(0, self.uniform_buffer2.clone())],
            [],
        )
        .unwrap();

        // Begin render pipeline commands
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    // clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    clear_values: vec![
                        Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0])),
                        Some(ClearValue::Depth(1.0)),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer)
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        // Render first draw pass
        let mut secondary_builder = AutoCommandBufferBuilder::secondary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [extent[0] as f32, extent[1] as f32],
            depth_range: 0.0..=1.0,
        };
        secondary_builder
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .bind_index_buffer(self.index_buffer.clone())
            .unwrap()
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set2,
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer2.clone())
            .unwrap()
            .bind_index_buffer(self.index_buffer2.clone())
            .unwrap()
            .draw_indexed(self.index_buffer2.len() as u32, 1, 0, 0, 0)
            .unwrap();
        let cb = secondary_builder.build().unwrap();
        builder.execute_commands(cb).unwrap();

        // Move on to next subpass for gui
        builder
            .next_subpass(
                Default::default(),
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();
        // Draw gui on subpass
        let cb = gui.draw_on_subpass_image([extent[0], extent[1]]);
        builder.execute_commands(cb).unwrap();

        // Last end render pass
        builder.end_render_pass(Default::default()).unwrap();
        let command_buffer = builder.build().unwrap();
        let after_future = before_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap();

        after_future.boxed()
    }
}

#[repr(C)]
#[derive(BufferContents, Vertex)]
struct MyVertex {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    color: [f32; 4],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

layout(location = 0) out vec4 v_color;
void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

    // gl_Position = vec4(position, 1.0);
    v_color = color;
}"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec4 v_color;

layout(location = 0) out vec4 f_color;

void main() {
    f_color = v_color;
}"
    }
}
