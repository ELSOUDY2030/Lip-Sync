from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

from scipy import signal
import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')


parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')



args = parser.parse_args()
args.img_size = 96




def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        face_det_results = face_detect([frames[0]]) 
    else:
        print('Using the specified bounding box instead of face detection...####################')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        
    idx = 0
    frame_to_save = frames[idx].copy()
    face, coords = face_det_results[idx].copy()
    batch_size = args.wav2lip_batch_size
    face = cv2.resize(face, (args.img_size, args.img_size))
    
    num_batches = int(np.ceil(len(mels) / batch_size))
    for i in range(num_batches):
        if i < num_batches - 1:
            batch_mels = mels[i * batch_size: (i + 1) * batch_size]
            batch_size_actual = batch_size
        else:
            batch_mels = mels[i * batch_size:]
            batch_size_actual = len(batch_mels)

        img_batch.extend([face] * batch_size_actual)
        frame_batch.extend([frame_to_save] * batch_size_actual)
        coords_batch.extend([coords] * batch_size_actual)

        img_masked = np.array(img_batch)
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, np.array(img_batch)), axis=3) / 255.
        mel_batch = np.reshape(np.array(batch_mels), [len(batch_mels), batch_mels[0].shape[0], batch_mels[0].shape[1], 1])


        yield img_batch, mel_batch, frame_batch, coords_batch
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def melspectrogram(wav):
    
    k=0.97
    ref_level_db = 20
    sample_rate = 16000
    n_fft=800
    num_mels=80
    fmin=55
    fmax=7600
    min_level_db=-100
    max_abs_value=4.0
    win_size=800
    hop_size = 200

    y = signal.lfilter([1, -k], [1], wav)
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_size, win_length=win_size)
    _mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels,fmin=fmin, fmax=fmax)

    min_level = np.exp(min_level_db / 20 * np.log(10))
    S = (20 * np.log10(np.maximum(min_level, np.dot(_mel_basis, np.abs(D))))) - ref_level_db
    _normalize = np.clip((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value,
                         -max_abs_value, max_abs_value)
    
    return _normalize

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to image file')

    elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        raise ValueError('--face argument must be a image file')

    print ("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = load_wav(args.audio, 16000)
    mel = melspectrogram(wav)
    print(mel.shape)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("####Length of mel chunks: {}#####".format(len(mel_chunks)))

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
                                            total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            model = load_model(args.checkpoint_path)
            print ("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                    cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

if __name__ == '__main__':
    main()
