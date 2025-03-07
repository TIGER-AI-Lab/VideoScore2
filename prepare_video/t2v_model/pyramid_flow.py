import os
from tqdm import tqdm
import requests
import replicate
from datetime import datetime

def run_pyramid_flow(prompts:list,raw_video_dir:str,video_names:list=[],
                    api_kwargs:dict={},):
    err_log=f"./api_err_report/pyramid_flow.txt"
    
    if "token" not in api_kwargs.keys():
        raise ValueError("arg error: api_kwargs-token")
    
    if "duration" not in api_kwargs.keys():
        api_kwargs["duration"] = 4
    if api_kwargs["duration"] not in range(1,10+1):
        raise ValueError("arg error: api_kwargs-duration")
    
    if "guidance_scale" not in api_kwargs.keys():
        api_kwargs["guidance_scale"] = 9
    if api_kwargs["guidance_scale"] not in range(1,15+1):
        raise ValueError("arg error: api_kwargs-guidance_scale")
    
    if "frames_per_second" not in api_kwargs.keys():
        api_kwargs["frames_per_second"] = 24
    if api_kwargs["frames_per_second"] not in [8,24]:
        raise ValueError("arg error: api_kwargs-frames_per_second")
        
    
    os.environ["REPLICATE_API_TOKEN"]=api_kwargs["token"]
    
    for idx,prompt in tqdm(enumerate(prompts)):
        video_path=os.path.join(raw_video_dir,f"{idx}.mp4")
        if video_names is not None and len(video_names)!=0:
            video_path=os.path.join(raw_video_dir,f"{video_names[idx]}.mp4")
        input={
            "prompt": prompt,
            "duration": api_kwargs["duration"],
            "guidance_scale": api_kwargs["guidance_scale"],
            "frames_per_second": api_kwargs["frames_per_second"],
            "video_guidance_scale": 5
        }
        
        
        try:
            output = replicate.run(
                "zsxkib/pyramid-flow:8e221e66498a52bb3a928a4b49d85379c99ca60fec41511265deec35d547c1fb",
                input=input
            )
            response = requests.get(output)
            if response.status_code == 200:
                with open(video_path, "wb") as file:
                    file.write(response.content)
            else:
                raise ValueError("download error, status code:", response.status_code)
        except Exception as e:
            date_time = datetime.now().strftime("%m-%d--%H-%M-%S")
            if "download error" in str(e):
                err_text=f"{date_time}, Download error: {idx}-th prompt"
            else:
                err_text=f"{date_time}, API calling error: {idx}-th prompt"
                
            with open(err_log, "a") as file:
                file.write(err_text + "\n")
        


