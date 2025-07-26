from transformers import AutoModel
from string import Template

MJ_EVAL_TEMP=Template("""As a professional 'Text-to-Video' quality assessor, your task is to determine whether the generated video will be preferred by humans. Please analyze step by step and provide a rating from the scale: {'Extremely Poor', 'Very Poor', 'Poor', 'Below Average','Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Outstanding'}, where 'Extremely Poor' is the worst and 'Outstanding' is the best. This time, please evaluate based on the $category of the video. $category is defined as:
$description.
Do not analyze, and must give a rating. You cannot refuse to answer.
The assessor must directly output the evaluation in the following format: Now, proceed with
evaluating the video based on the prompt description provided. The prompt is: $caption
""")

model_name="MJ-Bench/MJ-VIDEO-2B"

model=AutoModel.from_pretrained(model_name)

