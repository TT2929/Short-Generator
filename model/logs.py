from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import seaborn as sns
from io import StringIO
import pandas as pd
import funny_model


def append_labeled_segments_to_log(segment_csv: str, selected_filename: str, log_path: str = "labeled_log.csv", log_video: str = "video_log.csv"):

    df = pd.read_csv(segment_csv)
    df["label"] = df["filename"].apply(lambda x: 1 if x == selected_filename else 0)
    df["video_id"] = funny_model.video_id

    sem_cols = [col for col in df.columns if col.startswith("sem_")]
    output_cols = ["video_id", "filename", "label"] + sem_cols
    df_log = df[output_cols]


    if os.path.exists(log_path):
        df_log.to_csv(log_path, mode="a", index=False, header=False)
    else:
        df_log.to_csv(log_path, mode="w", index=False, header=True)


    df["video_url"] = funny_model.create_URL 
    cols_video = ["video_id", "video_url"]
    df_log_video = df[cols_video].drop_duplicates(subset=["video_id"])

    if os.path.exists(log_video):
        df_log_video.to_csv(log_video, mode="a", index=False, header=False)
    else:
        df_log_video.to_csv(log_video, mode="w", index=False, header=True)


if __name__ == '__main__':

    base_directory = Path("")

    #output-folder
    output_folder = Path( base_directory / "Output" )
    output_folder.mkdir(exist_ok=True)


    current_segments_path = output_folder / "segment" / "all_segment_feature.csv"
    log_output_base = output_folder / "logs" 
    log_output_base.mkdir(exist_ok=True, parents=True)

    labeled_log_path = log_output_base / "labeled_log.csv"
    video_log_path = log_output_base / "video_log.csv"


    append_labeled_segments_to_log(
        current_segments_path,       
        selected_filename="Target_cut_transcription_3.csv",            
        log_path=str(labeled_log_path),             
        log_video=str(video_log_path) 
    )
