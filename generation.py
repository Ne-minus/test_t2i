import torch
import huggingface_hub
from dataset_sampler import GenerationDataset
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, Transformer2DModel, PixArtSigmaPipeline, StableDiffusion3Pipeline, DiffusionPipeline, HunyuanDiTPipeline
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml

with open(r"./t2i_configs.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)


huggingface_hub.login(token='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')


class InitializeModels:
    def __init__(self, model_name, outputdir_name):
        self.model_path = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA is not available")
        dir_name = model_name.replace('/', '_')

        self.dir  = f"{outputdir_name}/{dir_name}"
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)

        self.pipe_def()
        
    def pipe_def(self):

        if self.model_path == 'stabilityai/stable-diffusion-3-medium-diffusers':
                        self.pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path,
                                                                 torch_dtype=torch.float16,
                                                                 variant="fp16")

        elif self.model_path == 'stabilityai/stable-diffusion-xl-base-1.0' or  self.model_path == 'playgroundai/playground-v2.5-1024px-aesthetic':
            self.pipe = DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)

        elif self.model_path == 'runwayml/stable-diffusion-v1-5' or self.model_path == 'prompthero/openjourney':
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path, safety_checker = None, torch_dtype=torch.float16)
            
        elif self.model_path == 'stabilityai/sdxl-turbo' or self.model_path == 'kandinsky-community/kandinsky-3':
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=torch.float16, variant="fp16")

        elif self.model_path == 'PixArt-alpha/PixArt-Sigma-XL-2-512-MS':
            self.pipe = PixArtSigmaPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        )

        elif self.model_path == "DeepFloyd/IF-I-XL-v1.0":
            self.pipe = DiffusionPipeline.from_pretrained(self.model_path, variant="fp16", torch_dtype=torch.float16)
            self.pipe.enable_model_cpu_offload()

        elif self.model_path == "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers":
            self.pipe = HunyuanDiTPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)

        self.pipe.to(self.device)


    def generate_image(self, batch):
        images = self.pipe(list(batch[1]), 
                            num_inference_steps=28,
                            guidance_scale=7.0,).images
    
        for idx, pic in zip(batch[0].tolist(), images):
            pic.save(f"{self.dir}/{idx}.png")


if __name__ == '__main__':
    our_set = GenerationDataset(params_list["DATASET_PATH"][0])
    loader = DataLoader(our_set, batch_size=4)

    print(params_list)

    model = InitializeModels(params_list["MODEL_NAME"][0], params_list["OUTPUT_DIR"][0])

    for batch in tqdm(loader):
        model.generate_image(batch)
