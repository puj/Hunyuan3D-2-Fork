


import torch
from PIL import Image

import os
import sys
print(os.listdir('Hunyuan3D-2-Fork/'))
print(sys.path)  # Ensure the correct path is included

subdirectory_path = os.path.join(os.getcwd(), 'Hunyuan3D-2-Fork/')
sys.path.append(subdirectory_path)

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

base_path = 'Hunyuan3D-2-Fork/'

def image_to_3d(image_path=base_path + 'assets/demo.png'):
    rembg = BackgroundRemover()
    model_path = 'tencent/Hunyuan3D-2'

   # Load image from file
    image = Image.open(image_path)
    print('Image loaded')

    if image.mode == 'RGB':
        image = rembg(image)

    print('Loading 3d model')
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

    # Generate 3d model
    print('Generating 3d')
    mesh = pipeline(image=image, num_inference_steps=30, mc_algo='mc',
                    generator=torch.manual_seed(2025))[0]

    # Post processing
    print('Post processing')
    print('Removing floaters')
    mesh = FloaterRemover()(mesh)
    print('Removing degenerate faces')
    mesh = DegenerateFaceRemover()(mesh)
    print('Reducing faces')
    mesh = FaceReducer()(mesh)
    print('Exporting')
    mesh.export('mesh.glb')
    print('Mesh exported')


    textured_mesh = None
    try:
        # Load texture model
        print('Loading texture model')
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        print('Generating texture')
        textured_mesh = pipeline(mesh, image=image)
        textured_mesh.export('texture.glb')
        print('Texture generated')

    except Exception as e:
        print(e)
        print('Please try to install requirements by following README.md')

    return mesh, textured_mesh



def text_to_3d(prompt='a car'):
    print('Starting text to 3d')

    print('Loading text models')
    t2i = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')

    print('Text Models loaded')

    # Convert text to image
    print('Processing text')
    image = t2i(prompt)

    # Remove background
    print('Removing background')
    rembg = BackgroundRemover()
    image = rembg(image)

    # Debug, output image
    image.save('text.png')
    print('Image saved')



    mesh, textured_mesh = image_to_3d(image_path='text.png')
    return mesh, textured_mesh


if __name__ == '__main__':
    # image_to_3d()
    # mesh, textured_mesh= text_to_3d(prompt="The Shardbearer: A tall, ethereal figure with translucent, crystalline skin glowing faintly blue. Their veins shimmer like flowing rivers of light, and a jagged, cracked sword floats at their side, held by unseen forces. Their eyes burn with the essence of a dying star")
    # mesh, textured_mesh= text_to_3d(prompt="The Astral Wayfinder: A cosmic traveler with a cloak that appears to be stitched from the night sky, speckled with stars and galaxies. Their boots leave glowing constellations in their footsteps, and their weapon, a crescent-shaped blade, radiates lunar light.")
    # mesh, textured_mesh= text_to_3d(prompt="The Gilded Revenant: A reanimated knight whose body is an intricate patchwork of gleaming gold and tarnished bronze. One eye burns with a spectral green light, and their skeletal frame is adorned with ceremonial ribbons that flutter in an unnatural wind.")
    # mesh, textured_mesh= text_to_3d(prompt="The Aetherborne: A slender, floating figure with translucent, jellyfish-like appendages that trail behind them in the air. Their semi-transparent body shifts colors like an oil slick, and their hands constantly weave glowing, ephemeral threads.")
    mesh, textured_mesh= text_to_3d(prompt="Chronoak of Ages: A massive, weathered tree with rings of glowing, shifting patterns on its trunk, each representing a different epoch in time. Its branches twist and form intricate, labyrinthine networks above.")

    # Ucomment to download image
    from google.colab import files
    files.download('text.png')
    files.download('texture.glb')

    print('Done')
