import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "utils_q_align")))
from q_align import QAlignVideoScorer, load_video


class eval_Q_Align:
    def __init__(self):
        self.model = QAlignVideoScorer()

    def evaluate_video(
        self,
        user_prompt: str,  
        video_path: str,
        kwargs: dict
    ) -> tuple[float, float, float, str]:
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")
        
        # optional: allow override frame count or fps
        infer_fps = kwargs.get("infer_fps", 2.0)
        try:
            video = load_video(video_path, fps=infer_fps)
        except Exception as e:
            print(e)
            return None, None, None, None

        try:
            score = self.model([video]).tolist()[0]
        except Exception as e:
            print(e)
            return None, None, None, None

        return score, score, score, None