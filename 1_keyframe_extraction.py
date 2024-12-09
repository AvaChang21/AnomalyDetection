import os
import shutil
from tqdm import trange, tqdm

import cv2
import numpy as np

'''
1.  查看画面是否存在运动光斑，设定阈值，超过阈值证明该画面重要，
    认为从此重要画面之后到没有光斑的画面存在动作反馈，截取有动作反馈的视频部分
2.  对比1得到的视频中前后两个frame的运动光斑是否移动，
    即运动光斑前后是否有位置变化(设定阈值，超过阈值认为该画面有物体运动)，
    来提取关键帧

Test:
    Normal 010/051/100/
    
Train:
    RoadAccidents 071
    Stealing 072
    Burglary 039
    Normal 916/942/295/271/664/211/646/
'''


def extract_start_end(cap, start_threshold=100, end_threshold=50):
    '''
    Part I
    :param cap: 
    :param start_threshold: 开始帧的运动斑点数量阈值
    :param end_threshold: 结束帧的运动斑点数量阈值
    :param min_gap: start_frame 和 end_frame 的最小帧数间隔
    :return: 返回开始和结束帧的索引
    '''

    start_frame = None
    end_frame = None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_gap = int(total_frames / 3)

    if not cap.isOpened():
        print("Unable to Open Video File")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Unable to Read the First Frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    frame_idx = 0
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_diff = cv2.absdiff(prev_gray, gray)

        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        motion_spots = np.sum(thresh)

        if start_frame is None and motion_spots > start_threshold:
            start_frame = frame_idx

        if start_frame is not None and motion_spots <= end_threshold:
            stable_count += 1
            if stable_count >= 5 and (frame_idx - start_frame) >= min_gap:
                end_frame = frame_idx
                break
        else:
            stable_count = 0

        prev_gray = gray
        frame_idx += 1

    if start_frame is not None and end_frame is not None:
        pass
    else:
        start_frame = 0
        end_frame = total_frames

    return start_frame, end_frame


def video_segment(video_path, output_path):
    '''
    Part I - 2th
    :param video_path:
    :param output_path:
    :return:
    '''
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame, end_frame = extract_start_end(cap)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count <= end_frame and count >= start_frame:
            out.write(frame)
        if count > end_frame:
            break

    cap.release()
    out.release()


def keyframes_by_motion(cap, keyframe_dir, blob_dir, movement_threshold=10):
    '''
    Part II
    :param cap:
    :param keyframe_dir:
    :param blob_dir:
    :param movement_threshold: 运动阈值
    :return:
    '''

    if not cap.isOpened():
        print("Unable to Open Video File")
        return

    ret, prev_frame = cap.read() 
    if not ret:
        print("Unable to Read the First Frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    keyframes = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_diff = cv2.absdiff(prev_gray, gray)

        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movement_detected = False

        # calculate centroids
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                if 'prev_cX' in locals():
                    distance = np.sqrt((cX - prev_cX) ** 2 + (cY - prev_cY) ** 2)

                    if distance > movement_threshold:
                        keyframes.append(frame_idx)
                        movement_detected = True
                        cv2.imwrite(os.path.join(keyframe_dir, f'frame_{frame_idx}.jpg'), frame)
                        cv2.imwrite(os.path.join(blob_dir, f'frame_{frame_idx}.jpg'), thresh)
                        break

                prev_cX, prev_cY = cX, cY

        if movement_detected:
            prev_gray = gray
        frame_idx += 1

    cap.release()
    return keyframes


def keyframes_by_motion1(cap, keyframe_dir, blob_dir, movement_threshold=10, min_frames=8):
    '''
    Part II
    :param cap:
    :param keyframe_dir:
    :param blob_dir:
    :param movement_threshold: 运动阈值
    :param min_frames
    :return:
    '''

    if not cap.isOpened():
        print("Unable to Open Video File")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    ret, prev_frame = cap.read()
    if not ret:
        print("Unable to Read the First Frame")
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    keyframes = []
    frame_idx = 1

    prev_cX, prev_cY = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_diff = cv2.absdiff(prev_gray, gray)

        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        movement_detected = False

        # calculate centroids
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])

                if prev_cX is not None and prev_cY is not None:
                    distance = np.sqrt((cX - prev_cX) ** 2 + (cY - prev_cY) ** 2)

                    if distance > movement_threshold:
                        keyframes.append(frame_idx)
                        movement_detected = True
                        cv2.imwrite(os.path.join(keyframe_dir, f'frame_{frame_idx}.jpg'), frame)
                        cv2.imwrite(os.path.join(blob_dir, f'frame_{frame_idx}.jpg'), thresh)
                        break

                prev_cX, prev_cY = cX, cY

        if movement_detected:
            prev_gray = gray
        frame_idx += 1

    # less than min_frames threshold
    if len(keyframes) < min_frames:
        step = fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

            frame_idx += 1
            if total_frames < min_frames:
                keyframes.append(frame_idx)
                cv2.imwrite(os.path.join(keyframe_dir, f'frame_{frame_idx}.jpg'), frame)
                cv2.imwrite(os.path.join(blob_dir, f'frame_{frame_idx}.jpg'), thresh)
            if min_frames < total_frames < (min_frames * step):
                if frame_idx % (step / 2) == 0:
                    keyframes.append(frame_idx)
                    cv2.imwrite(os.path.join(keyframe_dir, f'frame_{frame_idx}.jpg'), frame)
                    cv2.imwrite(os.path.join(blob_dir, f'frame_{frame_idx}.jpg'), thresh)
            else:
                if frame_idx % step == 0:
                    keyframes.append(frame_idx)
                    cv2.imwrite(os.path.join(keyframe_dir, f'frame_{frame_idx}.jpg'), frame)
                    cv2.imwrite(os.path.join(blob_dir, f'frame_{frame_idx}.jpg'), thresh)

    cap.release()
    return keyframes


v_dir = '/Users/changyichen/Desktop/MasterThesis/datasets/UCF_Crimes/Anomaly_Detection'
o_dir = '/Users/changyichen/Desktop/MasterThesis/datasets/UCF_Crimes/new_Segmented/'

folders = []

for _, t, _ in os.walk(v_dir):
    folders.extend(t)
    break

min_frames = 10
for tt in tqdm(folders, position=0, desc='Processing......', leave=False, ncols=80):
    video_dir = os.path.join(v_dir, tt)
    base_dir = os.path.join(o_dir, tt)
    try:
        os.makedirs(base_dir)
    except:
        pass
    dirs = []
    for _, ds, _ in os.walk(video_dir):
        dirs.extend(ds)
        break
    for dir_name in tqdm(dirs, position=1, desc=f'Processing Directories in {tt}', leave=False, ncols=80):
        video_dirs = os.path.join(video_dir, dir_name)
        for _, _, f in os.walk(video_dirs):
            for file in tqdm(f, position=2, desc=f'Processing files in {dir_name}', leave=False, ncols=80):
                video_path = os.path.join(video_dirs, file)
                out_video_dir = os.path.join(base_dir, dir_name, file.split('.')[0], 'video')
                keyframe_dir = os.path.join(base_dir, dir_name, file.split('.')[0], 'keyframe')
                blob_dir = os.path.join(base_dir, dir_name, file.split('.')[0], 'blob')

                try:
                    os.makedirs(blob_dir)
                    os.makedirs(keyframe_dir)
                    os.makedirs(out_video_dir)
                except:
                    pass

                output_path = os.path.join(out_video_dir, file)
                video_segment(video_path, output_path)

                cap = cv2.VideoCapture(output_path)
                keyframes = keyframes_by_motion1(cap, keyframe_dir, blob_dir, min_frames=min_frames)

                os.remove(output_path)

                shutil.rmtree(out_video_dir)
