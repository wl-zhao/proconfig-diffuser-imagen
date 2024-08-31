import torch
from pydantic import Field

from proconfig.widgets.base import WIDGETS
from proconfig.widgets.imagen_widgets.base import ImagenBaseWidget
from proconfig.widgets.imagen_widgets.utils.custom_types import IMAGE

from diffusers import DiffusionPipeline, UniPCMultistepScheduler

from torchvision.transforms.functional import to_tensor

@WIDGETS.register_module()
class DiffuserImagenWidget(ImagenBaseWidget):
    CATEGORY = "Diffusers/Image Generation"
    NAME = "Diffusers Image Generation"
    
    class InputsSchema(ImagenBaseWidget.InputsSchema):
        model_id: str = Field("stabilityai/stable-diffusion-xl-base-1.0", description="the huggingface model id")
        prompt: str = Field(..., description="the positive prompt")
        negative_prompt: str = Field("", description="the negative prompt")
        num_inference_steps: int = 15
    
    class OutputsSchema(ImagenBaseWidget.OutputsSchema):
        image: IMAGE
        
    def execute(self, environ, config):
        pipe = DiffusionPipeline.from_pretrained(config.model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        
        images = pipe(prompt=config.prompt, negative_prompt=config.negative_prompt, num_inference_steps=config.num_inference_steps)['images']
        image = torch.stack([to_tensor(im) for im in images]).permute(0, 2, 3, 1)

        return {
            "image": image
        }
        
        
if __name__ == "__main__":
    widget = DiffuserImagenWidget()
    config = {
        "prompt": "An astronaut riding a green horse, photo-realistic",
        "negative_prompt": "bad quality"
    }
    output = widget({}, config)
    import pdb; pdb.set_trace()