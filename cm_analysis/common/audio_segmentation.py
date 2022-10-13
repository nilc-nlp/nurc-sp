import os
import logging

import concurrent.futures

import librosa
import soundfile as sf
from tqdm import tqdm


def segment_raw_audios(args, corpus_sentences):
    def process(audio_path):
        logging.info(f"Segmenting audio {audio_path}")
        sample = os.path.splitext(os.path.basename(audio_path))[0]
        full_audio_path = os.path.abspath(audio_path)
        
        if args.load_full_audio:
            # Carrega o áudio inteiro na memória antes (economiza tempo no for seguinte)
            logging.info(f"Loading audio {full_audio_path}")
            full_audio, _ = librosa.load(full_audio_path, sr=args.sample_rate)
        else:
            full_audio = None

        for r in tqdm(corpus_sentences[sample]):
            start_sec = r["start_sec"]
            end_sec = r["end_sec"]
            duration = r["duration"]
                                        
            sentence_audio_path = os.path.join(
                args.audio_out_dir, 
                f"{sample}_{start_sec}_{end_sec}.wav"
            )

            if os.path.isfile(sentence_audio_path) and not args.overwrite_audios_dir:
                logging.debug(f"The segmented audio {sentence_audio_path} exists. Skipping segmentation.")
                continue

            if full_audio is not None:
                audio = full_audio[int(args.sample_rate*start_sec):int(args.sample_rate*end_sec)]
            else:
                audio, _ = librosa.load(full_audio_path,
                                        sr=int(args.sample_rate), 
                                        offset=start_sec, 
                                        duration=duration)
            sf.write(sentence_audio_path, audio, int(args.sample_rate))
        

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     futures = [executor.submit(process, audio_path) for audio_path in args.audio_files]
    #     for f in tqdm(concurrent.futures.as_completed(futures)):
    #         pass

    for audio_path in args.audio_files:
        process(audio_path)
