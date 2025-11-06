from gradio_client import Client, handle_file
import traceback

def wan22(prompt, img_path):
  client = Client("url")
  try:
    result = client.predict(
      prompt=prompt,
      negative_prompt='',
      seed=1,
      steps=4,
      input_image=handle_file(img_path),
      end_image=None,
      mode_selector="图生视频"，
      fps_slider=24,
      input_video=None,
      prompt_refiner=False,
      lora_selector=[],
      height=720,
      width=1280,
      frame_num=81,
      api_name="/generate_video"
    )
    print(result)
  except Exception as e:
    print(f"{str(e)}")
    trceback.print_exc()
