from yt_dlp import YoutubeDL
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from moviepy.editor import VideoFileClip, AudioFileClip
import csv
import json
import re
import seaborn as sns
from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
from pydub import AudioSegment
import whisper
import datetime
from spleeter.separator import Separator
import shutil
from collections import Counter
from sentence_transformers import SentenceTransformer
import joblib
import streamlit as st
import model_st
from multiprocessing import Pool, cpu_count
import glob




def create_movie(output_folder, create_URL):

    ydl_opts = {
        "format": "best",
        "writesubtitles" : True,
        "outtmpl": str(output_folder / "Target.%(ext)s")
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([create_URL])

def create_chat_csv(input_file_path, output_folder):
    output_file_path = output_folder / (Path(input_file_path).stem + '.csv')
    extracted_data = []

    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for line in lines[1:]:
        try:
            json_data = json.loads(line.strip())
            message = json_data['replayChatItemAction']['actions'][0]['addChatItemAction']['item']['liveChatTextMessageRenderer']['message']['runs'][0]['text']
            name = json_data['replayChatItemAction']['actions'][0]['addChatItemAction']['item']['liveChatTextMessageRenderer']['authorName']['simpleText']
            time = json_data['replayChatItemAction']['actions'][0]['addChatItemAction']['item']['liveChatTextMessageRenderer']['timestampText']['simpleText']

            extracted_data.append({'message': message, 'Name': name, 'time': time})
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"ERROR: {e}, row: {line.strip()}")
            continue

    with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['message', 'Name', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extracted_data)

    print("Create CSV chat file")


def funny_point(input_csv_file, output_folder):
    try:
        df = pd.read_csv(input_csv_file)

        if "time" in df.columns:
            def convert_time_to_seconds(time_str):
                try:
                    if isinstance(time_str, (int, float)):
                        return time_str

                    parts = time_str.split(':')
                    if len(parts) == 2: # MM:SS
                        minutes = int(parts[0])
                        seconds = int(parts[1])
                        total_seconds = minutes * 60 + seconds
                    elif len(parts) == 3: # HH:MM:SS
                        hours = int(parts[0])
                        minutes = int(parts[1])
                        seconds = int(parts[2])
                        total_seconds = hours * 3600 + minutes * 60 + seconds
                    else:
                        return None 

                    if time_str.startswith('-'):
                        return -total_seconds
                    return total_seconds
                except ValueError:
                    return None 

            df['time_in_seconds'] = df['time'].astype(str).apply(convert_time_to_seconds)
            df = df.dropna(subset=['time_in_seconds']) 

        else:
            print("WARNING: it do not exist time")
            df['time_in_seconds'] = None

        print("COMPLETE: Time to sec")
        print(df.head())

    except FileNotFoundError:
        print(f"ERROR: it do not exist csv_file: {input_csv_file}")
        exit()
    except Exception as e:
        print(f"ERROR: {e}")
        exit()

    funny_keywords = [r"笑", r"笑+" , r"笑笑笑+", r"草", r"www+", r"w+", r"おもしろ", r"おもろ", r"面白"]
    keyword_pattern = "|".join(funny_keywords)
    df["is_funny"] = df["message"].astype(str).apply(
        lambda x: bool(re.search(keyword_pattern, x, re.IGNORECASE))
    )
    funny_comments_df = df[df["is_funny"]].copy()

    if funny_comments_df.empty:
        print("\nit can not find funny keywords")


    bin_size_seconds = 10
    min_time = df['time_in_seconds'].min()
    max_time = df['time_in_seconds'].max()
    bins = range(int(min_time), int(max_time) + bin_size_seconds, bin_size_seconds)

    funniness_counts_over_time = pd.cut(
        funny_comments_df['time_in_seconds'], bins=bins, right=False
    ).value_counts().sort_index().fillna(0)

    bin_labels = [interval.left for interval in funniness_counts_over_time.index]

    plot_data = pd.Series(funniness_counts_over_time.values, index=bin_labels)
    plt.figure(figsize=(15, 7))
    sns.lineplot(x=plot_data.index, y=plot_data.values, marker='o', color='green')

    plt.title('funny_coment_counts', fontsize=16)
    plt.xlabel('time of video', fontsize=12)
    plt.ylabel('counts', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    if len(plot_data.index) > 10: 
        step = int(len(plot_data.index) / 10) * bin_size_seconds
        plt.xticks(range(int(min_time), int(max_time) + step, step))

    plt.tight_layout()
    output_pdf_path = output_folder / (Path(input_csv_file).stem + '.pdf') 
    plt.savefig(output_pdf_path, format="pdf")

    threshold_count = plot_data.quantile(0.85) 
    hot_segments_counts = plot_data[plot_data >= threshold_count]

    if not hot_segments_counts.empty:
        max_comment_count = hot_segments_counts.max()
        most_popular_start_sec = hot_segments_counts.idxmax()
        most_popular_end_sec = most_popular_start_sec + bin_size_seconds

    return most_popular_start_sec


def mp4_to_mp3(file_path):
    if file_path:
        try:
            video_clip = VideoFileClip(str(file_path))

            if video_clip.audio:
                output_path = os.path.splitext(file_path)[0] + ".mp3"
                video_clip.audio.write_audiofile(output_path)

                print("create mp3")
            else:
                print("WARNING: it do not exist audio clip")

        except Exception as e:
            print(f"ERROR: {e}")

    else:
        print("WARNING: it do not exist file")


def cut_mp3(i, segment, input_mp3_file, output_folder):
    audio = AudioFileClip(str(input_mp3_file))
    output_folder_cut = Path(output_folder) / "segment"
    output_folder_cut.mkdir(parents=True, exist_ok=True)

    cut_start_time_sec = segment[0]
    cut_end_time_sec = segment[1]

    clip = audio.subclip(cut_start_time_sec, cut_end_time_sec)
    output_audio_name = f"Target_cut_{i+1}.mp3" 
    output_path = output_folder_cut / output_audio_name
    clip.write_audiofile(str(output_path), codec='libmp3lame')


def cut_mp4(top_time, input_mp4_file, output_folder):
    video = VideoFileClip(str(input_mp4_file))
    cut_start_time_sec = top_time[0]
    cut_end_time_sec = top_time[1]

    clip = video.subclip(cut_start_time_sec, cut_end_time_sec)
    output_video_name = "Short_video.mp4" 
    output_path = output_folder / output_video_name
    clip.write_videofile(str(output_path))


def translate_mp3(i, segment, output_base_folder: str):

    output_folder = Path(output_base_folder) / "segment_translate"
    output_folder.mkdir(parents=True, exist_ok=True)
    input_cutmp3_path = Path(output_base_folder) / "segment" / f"Target_cut_{i+1}.mp3"
        
    try:
        audio = AudioSegment.from_mp3(input_cutmp3_path)
        print("COPMLETE:download MP3 file")
    except Exception as e:
        print(f"ERROR:{e}")
        return

    model = whisper.load_model("large-v3")

    all_segments_data = []
    temp_wav_path = output_folder / f"Target_cut_full_temp_{i+1}.wav"
    audio.export(temp_wav_path, format="wav")

    print(f"START TRANSLATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result = model.transcribe(str(temp_wav_path), language="ja")
    print(f"END TRANSLATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if temp_wav_path.exists():
        os.remove(temp_wav_path)

    full_transcription_text = result["text"]
    text_output_path = output_folder / f"Target_cut_full_text_{i+1}.txt"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription_text)

    if "segments" in result:
        for segment in result["segments"]:
            segment_start = segment["start"] 
            segment_end = segment["end"]     
            segment_text = segment["text"].strip()

            all_segments_data.append({
                "Start_Time_Sec": segment_start,
                "End_Time_Sec": segment_end,
                "Text": segment_text,
            })

    if all_segments_data:
        df = pd.DataFrame(all_segments_data)
        csv_output_path = output_folder / f"Target_cut_transcription_{i+1}.csv"
        df.to_csv(csv_output_path, index=False, encoding='utf-8')
        print(f"COMPLETE TARGET TRANSLATE CSV FILE: '{csv_output_path}'")
    else:
        print("it do not find segment")


def transcribe_mp3_segments_to_csv(file_path: str, output_base_folder: str):
    file_name_without_ext = Path(file_path).stem 
    
    output_folder = Path(output_base_folder) / file_name_without_ext
    output_folder.mkdir(parents=True, exist_ok=True)
    

    print("Load MP3 file...")
    try:
        audio = AudioSegment.from_mp3(file_path)
        print("COMPLETE:MP3 file download")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    #BGM
    separator = Separator('spleeter:2stems') 
    spleeter_input_path = output_folder / f"{file_name_without_ext}_spleeter_input.wav"
    spleeter_path = output_folder / f"{file_name_without_ext}_spleeter_input"
    audio.export(spleeter_input_path, format="wav")

    separator.separate_to_file(str(spleeter_input_path), str(output_folder))

    vocal_track_path = output_folder/ f"{file_name_without_ext}_spleeter_input" / 'vocals.wav' 
    if vocal_track_path.exists():
        audio = AudioSegment.from_wav(str(vocal_track_path))
        print("COMPLETE:get vocal file")
    else:
        print("ERROR: it do not find vocal file")

    if spleeter_input_path.exists():
        os.remove(spleeter_input_path)
    shutil.rmtree(spleeter_path) 

    model = whisper.load_model("large-v3")

    all_segments_data = []
    temp_wav_path = output_folder / f"{file_name_without_ext}_full_temp.wav"
    audio.export(temp_wav_path, format="wav")

    print(f"START TRANSLATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    result = model.transcribe(str(temp_wav_path), language="ja")
    print(f"END TRANSLATE: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if temp_wav_path.exists():
        os.remove(temp_wav_path)

    full_transcription_text = result["text"]
    text_output_path = output_folder / "Target_cut_full_text.txt"
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(full_transcription_text)


    if "segments" in result:
        for segment in result["segments"]:
            segment_start = segment["start"] 
            segment_end = segment["end"]     
            segment_text = segment["text"].strip()

            all_segments_data.append({
                "Start_Time_Sec": segment_start,
                "End_Time_Sec": segment_end,
                "Text": segment_text,
            })
    
    if all_segments_data:
        df = pd.DataFrame(all_segments_data)
        csv_output_path = output_folder / "Target_cut_transcription.csv"
        df.to_csv(csv_output_path, index=False, encoding='utf-8')
        print(f"COMPLETE TARGET TRANSLATE CSV FILE: '{csv_output_path}'")
    else:
        print("it do not find segment")


def generate_candidate(funny_time):
    start_time = funny_time - 30
    end_time = funny_time + 30
    segments = []
    segment_duration = [20, 30, 40, 50, 60]
    stride_sec = 5
    for time in segment_duration:  
        current = start_time          
        while current + time <= end_time:
            segments.append((current, current + time))
            current += stride_sec
    return segments



def extract_embedding_features(i, input_dir: str, output_folder):
    output_file_path = output_folder / "segment" / f"segment_feature_{i+1}.csv"
    results = []
    model_BERT = SentenceTransformer("sentence-transformers/LaBSE")
    for file in os.listdir(input_dir):
        if file.endswith(f"_{i+1}.csv"):
            file_path = os.path.join(input_dir, file)

            df = pd.read_csv(file_path)
            text = " ".join(df["Text"].astype(str).tolist())

            vec = model_BERT.encode(text)
            feature_dict = {f"sem_{i}": float(v) for i, v in enumerate(vec)}
            feature_dict["filename"] = file

            results.append(feature_dict)

    feature_df = pd.DataFrame(results)
    feature_df.to_csv(output_file_path, index=False)


def process_segment(args):
    i, segment, input_mp3_path, output_folder = args

    cut_mp3(i, segment, input_mp3_path, output_folder)

    #BGM処理を行わない場合
    translate_mp3(i, segment, output_folder)
    #BGMの分離を行う場合
    #transcribe_mp3_segments_to_csv(input_cutmp3_path, output_folder)

    input_trans_path = base_directory / "Output" / "segment_translate"
    extract_embedding_features(i, input_trans_path, output_folder)



def parallel_segment_process(segments, input_mp3_path, output_folder):
    Path(output_folder / "segment_translate").mkdir(parents=True, exist_ok=True)

    args_list = [(i, segment, input_mp3_path, output_folder) for i, segment in enumerate(segments)]

    with Pool(processes=len(segments)) as pool:
        pool.map(process_segment, args_list)


#Write URL that you want to have short movie
####################################################################################################
create_URL = "https://www.youtube.com/live/rQ8v_iyRs6Y?si=K_QZvef0mLvgmHU-"
video_id = "vid188"
####################################################################################################

if __name__ == '__main__':
    #base_directory
    base_directory = Path("/home/kondo.hayate/program/Short-Generator")

    #output-folder
    output_folder = Path( base_directory / "Output" )
    output_folder.mkdir(exist_ok=True)
    

    create_movie(output_folder, create_URL)
    input_mp4_path = base_directory / "Output" / "Target.mp4"
    mp4_to_mp3(input_mp4_path)


    funny_sec_all = []
    for _ in range(5):
        input_json_path = base_directory / "Output" / "Target.live_chat.json"
        create_chat_csv(input_json_path, output_folder)

        input_csv_path = base_directory / "Output" / "Target.live_chat.csv"
        funny_st_sec = funny_point(input_csv_path, output_folder)
        print(f"funny_st_sec:{funny_st_sec}")
        funny_sec_all.append(funny_st_sec)

    counts = Counter(funny_sec_all)
    most_common_element = counts.most_common(1)
    most_funny_st_sec = most_common_element[0][0]


    #model------------------
    segments = generate_candidate(most_funny_st_sec)

    input_mp3_path = base_directory / "Output" / "Target.mp3"

    parallel_segment_process(segments, input_mp3_path, output_folder)

    all_dfs = []
    for path in glob.glob(str(output_folder / "segment" / "segment_feature_*.csv")):
        df = pd.read_csv(path)
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(output_folder / "segment" / "all_segment_feature.csv", index=False)

    #-----judge funny-----
    SEGMENT_FEATURES = str(base_directory / "Output" / "segment" / "all_segment_feature.csv")
    model_path = base_directory / "model" / "funny_pred_model.pkl"
    df = pd.read_csv(SEGMENT_FEATURES)
    clf = joblib.load(str(model_path))

    X = df[[col for col in df.columns if col.startswith("sem_")]]
    df["score"] = clf.predict_proba(X)[:, 1]
    df_sorted = df.sort_values("score", ascending=False)

    top_segment = df_sorted.iloc[0]
    top_filename = top_segment["filename"]
    top_filename_ext = os.path.splitext(top_filename)[0]

    match = re.search(r"_(\d+)$", top_filename_ext)
    if match:
        number = int(match.group(1))
        print(f"Number: {number}")
    else:
        print("ERROR:it do not find number")

    top_time = segments[number - 1]    
    cut_mp4(top_time, input_mp4_path, output_folder)

    VIDEO_FILE = str(output_folder / "Short_video.mp4")
    LOG_FILE = str(output_folder / "logs" / "labeled_log.csv")

    st.title("funny judge")

    if os.path.exists(VIDEO_FILE):
        st.video(VIDEO_FILE)
    else:
        st.error("ERROR;it do not find video file")

    selected = df[df["filename"] == top_filename].copy()
    score_threshold = 0.5

    if selected.empty:
        st.error("it do not find segment")
    else:
        score = selected["score"].values[0]
        label = 1 if score >= score_threshold else 0

        selected["label"] = label
        selected["video_id"] = video_id

        sem_cols = [col for col in df.columns if col.startswith("sem_")]
        output_cols = ["video_id", "filename", "label"] + sem_cols
        selected = selected[output_cols]

        if os.path.exists(LOG_FILE):
            df_log = pd.read_csv(LOG_FILE)
            df_log = pd.concat([df_log, selected], ignore_index=True)
        else:
            df_log = selected
        
    df_log.to_csv(LOG_FILE, index=False)
    print(f"top_segment score = {score:.3f}, label = {label}")
    print("COMPLETE: it record funny features")

    video_log_path = base_directory/ "Output" / "logs" / "video_log.csv"
    log_video=str(video_log_path) 
    df["video_url"] = create_URL
    df["video_id"] = video_id
    cols_video = ["video_id", "video_url"]
    df_log_video = df[cols_video].drop_duplicates(subset=["video_id"])

    if os.path.exists(log_video):
        df_log_video.to_csv(log_video, mode="a", index=False, header=False)
    else:
        df_log_video.to_csv(log_video, mode="w", index=False, header=True)

    model_st.study_model()
    
    if input_mp4_path.exists():
        os.remove(input_mp4_path)
    if input_mp3_path.exists():
        os.remove(input_mp3_path)
    input_json_path = base_directory / "Output" / "Target.live_chat.json"
    if input_json_path.exists():
        os.remove(input_json_path)
    live_chat_csv = output_folder / "Target.live_chat.csv"
    if live_chat_csv.exists():
        os.remove(live_chat_csv)
    live_chat_pdf = output_folder / "Target.live_chat.pdf"
    if live_chat_pdf.exists():
        os.remove(live_chat_pdf)
    segment_path = output_folder / "segment"
    if segment_path.exists():
        shutil.rmtree(segment_path) 
    segment_trans_path = output_folder / "segment_translate"
    if segment_trans_path.exists():
        shutil.rmtree(segment_trans_path) 