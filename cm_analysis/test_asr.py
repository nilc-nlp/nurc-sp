import os
import json
import logging

import jiwer 
import textgrid
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from transformers import (
    pipeline, 
    AutomaticSpeechRecognitionPipeline, 
    Wav2Vec2FeatureExtractor, 
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2CTCTokenizer
)

from common import parse_textgrids, audio_segmentation, audio_preprocessing


def main(args):
    logging.info("Analysing TextGrid files")
    corpus_sentences, corpus_new_textgrids = parse_textgrids.parse_textgrids(args)
    logging.info("Starting audio segmentation")
    audio_segmentation.segment_raw_audios(args, corpus_sentences)
    run_test(args, corpus_sentences, corpus_new_textgrids)

def run_test(args, corpus_sentences, corpus_new_textgrids):
    logging.info("Starting tests...")
    test_results = {}
    summary = []

    logging.info(f"Loading model {args.model}")
    asr = pipeline(model=args.model, device=args.device)
    if args.phone_model is not None:  # TODO: this is a workaround to get the model to work
        feature_extractor =  Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-large-960h"
        )
        processor = Wav2Vec2Processor.from_pretrained(args.phone_model)
        phone_model = Wav2Vec2ForCTC.from_pretrained(args.phone_model)
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.phone_model)
        phone_model = AutomaticSpeechRecognitionPipeline(model=phone_model, tokenizer=tokenizer, feature_extractor=feature_extractor, processor=processor, device=args.device)

    pre_process_audio = audio_preprocessing.get_preprocessing_function(args.model, args.sample_rate)

    for audio_file in tqdm(args.audio_files):
        sample = os.path.splitext(os.path.basename(audio_file))[0]
        logging.info(f"Processing audio {sample}")
        
        predictions = []
        sentences = []
        wers = cers = mers = wils = 0
        test_results[sample] = []

        timestamps_word_dicts = []
        timestamps_char_dicts = []

        sample_start_sec = corpus_new_textgrids[sample].minTime
        sample_end_sec = corpus_new_textgrids[sample].maxTime

        transcription_tier = textgrid.IntervalTier(
            name="Transcription",
            minTime=sample_start_sec,
            maxTime=sample_end_sec
        )
        timestamps_word_tier = textgrid.IntervalTier(
            name="TimestampsWords",
            minTime=sample_start_sec,
            maxTime=sample_end_sec
        )
        timestamps_char_tier = textgrid.IntervalTier(
            name="TimestampsChar",
            minTime=sample_start_sec,
            maxTime=sample_end_sec
        )
        if args.phone_model is not None:
            phones_tier = textgrid.IntervalTier(
                name="Phones",
                minTime=sample_start_sec,
                maxTime=sample_end_sec
            )
            timestamps_phones_tier = textgrid.IntervalTier(
                name="TimestampsPhones",
                minTime=sample_start_sec,
                maxTime=sample_end_sec
            )

        for r in tqdm(corpus_sentences[sample], leave=False) if not args.log_level == "DEBUG" else corpus_sentences[sample]:
            start_sec = r["start_sec"]
            end_sec = r["end_sec"]
            duration = r["duration"]
                   
            sentence_audio_path = os.path.join(
                args.audio_out_dir, 
                f"{sample}_{start_sec}_{end_sec}.wav"
            )
            logging.debug(f"Preprocessing segment {sentence_audio_path}")
            audio = pre_process_audio(sentence_audio_path)
            if len(audio) < args.sample_rate:
                logging.debug(f"Adding padding to the segment {sentence_audio_path}")
                audio = np.pad(audio, 
                               pad_width=args.sample_rate, 
                               constant_values=0)  # Evita problemas de alocação se o áudio for muito curto
            sentence = r["text"]
            sentences.append(sentence)
            mark = r["mark"]

            if duration > args.max_duration:
                logging.debug(f"Maximum duration detected in the segment: {sentence_audio_path}"
                              f"({duration} seconds). Using windowing technique.")
                output_word_ts = asr(audio, 
                                     chunk_length_s=10, 
                                     stride_length_s=(4, 2), 
                                     return_timestamps="word")
                if args.generate_char_timestamps:  # TODO: we need to infer another time just to get the char timestamps?
                    output_char_ts = asr(audio, 
                                        chunk_length_s=10, 
                                        stride_length_s=(4, 2), 
                                        return_timestamps="char")
                    if args.phone_model is not None:
                        output_phones_ts = phone_model(audio, 
                                                    chunk_length_s=10, 
                                                    stride_length_s=(4, 2), 
                                                    return_timestamps="char")
            elif duration > 0:
                output_word_ts = asr(audio, 
                                     return_timestamps="word")
                if args.generate_char_timestamps:
                    output_char_ts = asr(audio, 
                                        return_timestamps="char")
                    if args.phone_model is not None:
                        output_phones_ts = phone_model(audio, 
                                                    return_timestamps="char")
                                     
            timestamps_word = output_word_ts["chunks"]
            if args.generate_char_timestamps:
                timestamps_char = output_char_ts["chunks"]
                timestamps_phones = output_phones_ts["chunks"]
           
            output = output_word_ts
            prediction = output["text"]
            predictions.append(prediction)

            if args.phone_model is not None and args.generate_char_timestamps:
                phones = output_phones_ts["text"]
            else:
                phones = ""

            if start_sec == sample_start_sec:
                start_sec += 0.000000000001
            if end_sec == sample_end_sec:
                end_sec -= 0.000000000001
            try:
                transcription_tier.add(minTime=start_sec, maxTime=end_sec, mark=prediction)
                if args.phone_model is not None:
                    phones_tier.add(minTime=start_sec, maxTime=end_sec, mark=phones)
            except Exception as e:
                logging.error(f"Could not add textgrid item ({r}): {str(e)}")

            try:
                logging.info(f"Calculating metrics {sentence_audio_path}")

                wer = cer = mer = wil = -1
                if 'all' in args.metrics or 'wer' in args.metrics:
                    wer = jiwer.wer(sentence, prediction)
                    wers += wer
                if 'all' in args.metrics or 'mer' in args.metrics:
                    mer = jiwer.mer(sentence, prediction)
                    mers += mer
                if 'all' in args.metrics or 'wil' in args.metrics:
                    wil = jiwer.wil(sentence, prediction)
                    wils += wil
                if 'all' in args.metrics or 'cer' in args.metrics:
                    cer = jiwer.cer(sentence, prediction)
                    cers += cer

                logging.debug(
                    f"{sentence_audio_path}:"
                    f"\n\tGT: {sentence}"
                    f"\n\tPR: {prediction}"
                    f"\n\tWER: {wer}"
                    f"\n\tMER: {mer}"
                    f"\n\tWIL: {wil}"
                    f"\n\tCER: {cer}"
                )
            except Exception as e:
                logging.error(f"Error calculating metrics {sentence_audio_path}: {str(e)}")
            
            test_results[sample].append({
                "path": sentence_audio_path,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "mark": mark,
                "sentence": sentence,
                "duration": end_sec-start_sec,
                "prediction": prediction,
                "phones": phones,
                "timestamps": timestamps_word,
                "wer": wer,
                "mer": mer,
                "wil": wil,
                "cer": cer
            })

            for tp in timestamps_word:
                if tp["text"].strip() == '':
                    logging.info(f"Empty timestamp word/char in {sentence_audio_path} at {tp['timestamp'][0]} sec")
                    continue
                ts_start_sec = start_sec+tp["timestamp"][0]
                ts_end_sec = start_sec+tp["timestamp"][1]
                timestamps_word_dicts.append({
                    "path": sentence_audio_path,
                    "tp_text": tp["text"],
                    "tp_start": tp["timestamp"][0],
                    "tp_end": tp["timestamp"][1],
                    "ts_start_sec": ts_start_sec,
                    "ts_end_sec": ts_end_sec
                })
                if ts_start_sec >= end_sec:
                    logging.debug(f"Timestamp start sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                    ts_start_sec = end_sec-0.000000000001
                    logging.debug(f"\tNew start sec: {ts_start_sec}")
                if ts_end_sec >= end_sec:
                    logging.debug(f"Timestamp end sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                    ts_end_sec = end_sec-0.000000000001
                    logging.debug(f"\tNew end sec: {ts_end_sec}")
                try:
                    timestamps_word_tier.add(minTime=ts_start_sec, maxTime=ts_end_sec, mark=tp["text"])
                except:
                    logging.error(f"Trying to add a tier: {timestamps_word_dicts[-1]}")

            if args.generate_char_timestamps:
                for tp in timestamps_char:
                    if tp["text"].strip() == '':
                        logging.info(f"Empty timestamp word/char in {sentence_audio_path} at {tp['timestamp'][0]} sec")
                        continue
                    ts_start_sec = start_sec+tp["timestamp"][0]
                    ts_end_sec = start_sec+tp["timestamp"][1]
                    timestamps_char_dicts.append({
                        "path": sentence_audio_path,
                        "tp_text": tp["text"],
                        "tp_start": tp["timestamp"][0],
                        "tp_end": tp["timestamp"][1],
                        "ts_start_sec": ts_start_sec,
                        "ts_end_sec": ts_end_sec
                    })
                    if ts_start_sec >= end_sec:
                        logging.debug(f"Timestamp start sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                        ts_start_sec = end_sec-0.000000000001
                        logging.debug(f"\tNew start sec: {ts_start_sec}")
                    if ts_end_sec >= end_sec:
                        logging.debug(f"Timestamp end sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                        ts_end_sec = end_sec-0.000000000001
                        logging.debug(f"\tNew end sec: {ts_end_sec}")
                    try:
                        timestamps_char_tier.add(minTime=ts_start_sec, maxTime=ts_end_sec, mark=tp["text"])
                    except:
                        logging.error(f"Trying to add a tier: {timestamps_char_dicts[-1]}")

                if args.phone_model is not None:
                    for tp in timestamps_phones:
                        if tp["text"].strip() == '':
                            logging.info(f"Empty timestamp word/char in {sentence_audio_path} at {tp['timestamp'][0]} sec")
                            continue
                        ts_start_sec = start_sec+tp["timestamp"][0]
                        ts_end_sec = start_sec+tp["timestamp"][1]
                        if ts_start_sec >= end_sec:
                            logging.debug(f"Timestamp start sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                            ts_start_sec = end_sec-0.000000000001
                            logging.debug(f"\tNew start sec: {ts_start_sec}")
                        if ts_end_sec >= end_sec:
                            logging.debug(f"Timestamp end sec ({ts_end_sec}) is higher than audio ({sentence_audio_path}) end sec: {end_sec}")
                            ts_end_sec = end_sec-0.000000000001
                            logging.debug(f"\tNew end sec: {ts_end_sec}")
                        try:
                            timestamps_phones_tier.add(minTime=ts_start_sec, maxTime=ts_end_sec, mark=tp["text"])
                        except Exception as e:
                            logging.error(f"Trying to add a timestamps_phones_tier: {str(e)}")

            sample_timestamps_word_pd = pd.DataFrame(timestamps_word_dicts)
            sample_timestamps_word_pd.to_csv(
                os.path.join(args.out_dir, f"{sample}_timestamps_word_dicts_{args.model.replace('/', '_')}.csv"),
                sep=';' if args.ptbr else ',',
                decimal=',' if args.ptbr else None,
                index=False,
                mode='a'
            )
            # Requires to much space!
            # sample_timestamps_char_pd = pd.DataFrame(timestamps_char_dicts)
            # sample_timestamps_char_pd.to_csv(
            #     os.path.join(args.out_dir, f"{sample}_timestamps_char_dicts_{args.model.replace('/', '_')}.csv"),
            #     sep=';' if args.ptbr else ',',
            #     decimal=',' if args.ptbr else None,
            #     index=False,
            #     mode='a'
            # )

        logging.info(f"{sample} audio successfully processed")

        try:
            logging.info(f"Calculating metrics of {sample}")
            if 'all' in args.metrics or 'wer' in args.metrics:
                total_sample_wer = jiwer.wer(sentences, predictions)
                wers += wer
            if 'all' in args.metrics or 'mer' in args.metrics:
                total_sample_mer = jiwer.mer(sentences, predictions)
            if 'all' in args.metrics or 'wil' in args.metrics:
                total_sample_wil = jiwer.wil(sentences, predictions)
            if 'all' in args.metrics or 'cer' in args.metrics:
                total_sample_cer = jiwer.cer(sentences, predictions)
        except Exception as e:
            logging.error(f"Unable to calculate metrics for {sample}: {str(e)}")

        total_audio_sentences = len(test_results[sample])
        
        if args.average_from_sentences:
            avg_sample_wer = wers/total_audio_sentences
            avg_sample_mer = mers/total_audio_sentences
            avg_sample_wil = wils/total_audio_sentences
            avg_sample_cer = cers/total_audio_sentences
            summary.append({
                "SAMPLE": sample,
                "AVG WER": avg_sample_wer,
                "AVG MER": avg_sample_mer,
                "AVG WIL": avg_sample_wil,
                "AVG CER": avg_sample_cer,
                "TOTAL WER": total_sample_wer,
                "TOTAL MER": total_sample_mer,
                "TOTAL WIL": total_sample_wil,
                "TOTAL CER": total_sample_cer
            })
            logging.info(f"AVG   WER {sample}: {avg_sample_wer}")
            logging.info(f"TOTAL WER {sample}: {total_sample_wer}")
            logging.info(f"AVG   MER {sample}: {avg_sample_mer}")
            logging.info(f"TOTAL MER {sample}: {total_sample_mer}")
            logging.info(f"AVG   WIL {sample}: {avg_sample_wil}")
            logging.info(f"TOTAL WIL {sample}: {total_sample_wil}")
            logging.info(f"AVG   CER {sample}: {avg_sample_cer}")
            logging.info(f"TOTAL CER {sample}: {total_sample_cer}")
        else:
            summary.append({
                "SAMPLE": sample,
                "WER": total_sample_wer,
                "MER": total_sample_mer,
                "WIL": total_sample_wil,
                "CER": total_sample_cer
            })
    
        logging.info(f"Exporting results of {sample}")
        sample_results_pd = pd.DataFrame(test_results[sample])
        sample_results_pd.to_csv(
            os.path.join(args.out_dir, f"{sample}_results_{args.model.replace('/', '_')}.csv"),
            sep=';' if args.ptbr else ',',
            decimal=',' if args.ptbr else None,
            index=False
        )
        if args.log_level in ('INFO', 'DEBUG'):
            print(f"Results of {sample}:")
            print(tabulate(
                sample_results_pd.drop("timestamps", axis=1), headers='keys', tablefmt='psql', maxcolwidths=45)
            )

        corpus_new_textgrids[sample].tiers.append(transcription_tier)
        corpus_new_textgrids[sample].tiers.append(timestamps_word_tier)
        corpus_new_textgrids[sample].tiers.append(timestamps_char_tier)
        if args.phone_model is not None:
            corpus_new_textgrids[sample].tiers.append(phones_tier)
            corpus_new_textgrids[sample].tiers.append(timestamps_phones_tier)
        logging.info(f"Exporting texgrid of {sample}: {os.path.join(args.out_dir, sample + '.TextGrid')}")
        corpus_new_textgrids[sample].write(os.path.join(args.out_dir, sample + '.TextGrid'))

    summary_pd = pd.DataFrame(summary)
    summary_pd.loc["AVG"] = summary_pd.mean()
    summary_pd.to_csv(
        os.path.join(args.out_dir, f"summary.csv"),  # os.path.join(args.out_dir, f"summary_{args.model.replace('/', '_')}.csv"),
        sep=';' if args.ptbr else ',',
        decimal=',' if args.ptbr else None,
        index=False
    )
    print(f"{'='*20} FINAL RESULTS {'='*20}")
    print(f"{args}")
    print(tabulate(summary_pd, headers='keys', tablefmt='psql', maxcolwidths=45))
    summary_pd = pd.DataFrame(summary)
    print(tabulate(summary_pd.describe(), headers='keys', tablefmt='psql', maxcolwidths=45))


def prepare_output_dirs(args):
    logging.info("Preparing output directories")

    logging.info("Creating output directory")
    if os.path.isdir(args.out_dir):
        logging.warning(f"The {args.out_dir} directory already exists. "
                         "The files will be overwritten.")
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info(f"Creating preprocessed audio directory: {args.audio_out_dir}")
    if os.path.isdir(args.audio_out_dir):
        logging.warning(f"The preprocessed audio directory {args.audio_out_dir} exists")
        if args.overwrite_audios_dir:
            logging.info(f"The {args.audio_out_dir} directory already exists. "
                          "Audio files will be overwritten.")
        else:
            logging.info(f"The {args.audio_out_dir} directory already exists. "
                          "Audio files will not be overwritten.")
    os.makedirs(args.audio_out_dir, exist_ok=True)


def check_files_list(args):
    assert len(args.audio_files) == len(args.textgrids), (
        "The list of audio files and textgrids must be the same size. "
        f"audio_files={len(args.audio_files)}, textgrids={len(args.textgrids)}"
    )
    
    for af, tf in zip(args.audio_files, args.textgrids):
        logging.info(f"Checking files {af} and {tf}")
        assert os.path.isfile(af), f"The file {af} does not exists"
        assert os.path.isfile(tf), f"The file {tf} does not exists"
        if os.path.splitext(os.path.basename(af))[0] != os.path.splitext(os.path.basename(tf))[0]:
            logging.warning("The names of the files passed are different. "
                            f"Is this correct? Áudio={af}, Texgrid={tf}")

def check_args(args):
    if args.accept_all and len(args.ignore_sentences_with) > 0:
        logging.warning("The accept_all argument is set to true, "
                        "but you passed arguments to ignore sentences: "
                        f"{args.ignore_sentences_with}. This argument will be discarded.")
        args.ignore_sentences_with = []
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("Teste de sentenças do NURC-SP. "
                                     "Forneça a lista de áudios e a lista de textgrids. "
                                     "A ordem deve ser respeitada. Ex: python test_asr.py audio1.wav "
                                     "audio1.TextGrid audio2.wav audio2.TextGrid ... audioN.wav audioN.TextGrid")
    parser.add_argument("--audio-files", "-f",
                        nargs="+", 
                        help="Arquivos de áudio para processar",
                        required=True)
    parser.add_argument("--textgrids", "-t",
                        nargs="+", 
                        help="Arquivos de textgrid para processar",
                        required=True)
    parser.add_argument("--only-ntb", 
                        help="Processa apenas as NTBs", 
                        action="store_true")
    parser.add_argument("--only-tb", 
                        help="Processa apenas as TBs", 
                        action="store_true")
    parser.add_argument("--out-dir", "-o",
                        help="Diretório de saída (logs, resultados, CSVs e JSONs)", 
                        default="./")
    parser.add_argument("--save-sentences-json-dir", 
                        help="Gera um único arquivo JSON contendo as sentenças de cada áudio")
    parser.add_argument("--save-skipped-sentences-json-dir",
                        help="Gera um único arquivo JSON contendo as sentenças ignoradas de cada áudio")
    parser.add_argument("--save-sentences-csv-dir", 
                        help="Gera os arquivos CSV contendo as sentenças de cada áudio")
    parser.add_argument("--save-skipped-sentences-csv-dir",
                        help="Gera os arquivos CSV contendo as sentenças ignoradas de cada áudio")
    parser.add_argument("--log-level", "-L",
                        help="Nível de log",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="WARNING")
    parser.add_argument("--log-file",
                        help="Arquivo de log (opcional)")
    parser.add_argument("--generate-word-timestamps", 
                        help="Generate word timespamps of transcribed audios", 
                        action="store_true")
    parser.add_argument("--generate-char-timestamps", 
                        help="Generate char timespamps of transcribed audios. Note: models with LM does not support this option.", 
                        action="store_true")
    text_norm_parser = parser.add_argument_group('Opções de normalização de texto')
    text_norm_parser.add_argument("--accept-all", 
                                  help="Aceita todas as sentenças. "
                                       "Remove apenas os caracteres que não fazem parte do alfabeto e os trechos"
                                       "com anotações sobre o áudio. Ex ((risos)).", 
                                  action="store_true")
    text_norm_parser.add_argument("--ignore-all",
                                  help="Ignora todas as sentenças com problemas (anotações, incompreensão, sobreposição, etc.), "
                                       "removendo os caracteres que não fazem parte do alfabeto.", 
                                  action="store_true")
    text_norm_parser.add_argument("--ignore-sentences-with", "-i" ,
                                  nargs="+", 
                                  help="Ignora sentenças dos tipos definidos:"
                                       "\nignore_hypothesis_sentences: trechos com incompreensão de palavras ou segmentos: "
                                       "anotado com \"(hipótese)\" ex:\"(em relação à profissão)\""
                                       "\nincomprehensible_sentences: trechos com incompreensão de palavras ou segmentos: "
                                       "anotado com \"( )\" ex:\"( )\""
                                       "\nsentences_with_annotation_parts:  trechos com anotação: "
                                       "anotado com \"(( ))\" ex:\"((risos))\""
                                       "\noverlap_sentences: trechos com sobreposição de fala anotado com \"[\""
                                       "\nPor exemplo: \"--ignore-sentences-with\" noverlap_sentences sentences_with_annotation_parts incomprehensible_sentences "
                                       "irá ignorar todos os casos."
                                       "\nCaso accept-all seja passado, esse argumento será automaticamente "
                                       "configurado com nenhuma das opções. Caso ignore-all seja passado, esse argumento será "
                                       "automaticamente configurado com todas as opções.",
                                  default=[])
    audio_parser = parser.add_argument_group('Opções de pré-processamento de áudio')
    audio_parser.add_argument("--audio-out-dir", 
                              help="Diretório de saída", 
                              default="./audios")
    audio_parser.add_argument("--sample-rate", '-sr',
                              help="Taxa de amostragem", 
                              default=16000)
    audio_parser.add_argument("--max-duration",
                              help="Máxima duração dos segmentos. Caso o segmento de áudio tenha uma duração maior "
                                   "que a definida, o áudio será transcrito usando a técnica de janelamento "
                                   "automaticamente", 
                              default=20)
    # audio_parser.add_argument("--skip-audio-segmentation", 
    #                           help="Ignora a etapa de pré-processamento de áudio", 
    #                           action="store_true")
    audio_parser.add_argument("--overwrite-audios-dir", 
                              help="Executa o pré-processamento dos áudios mesmo se a pasta de áudios existir. "
                                   "Caso o diretório de áudios exista e esta opção não for ativada, o "
                                   "pré-processamento será pulado automaticamente.", 
                              action="store_true")
    audio_parser.add_argument("--load-full-audio", 
                              help="Carrega todo o áudio para segmentar. Opção mais rápida, mas consome mais memória.", 
                              action="store_true")
    test_parser = parser.add_argument_group('Opções de teste')
    test_parser.add_argument("--model", "-m",
                             help="Path ou nome do modelo de ASR para realizar o teste", 
                             required=True)
    test_parser.add_argument("--phone-model", "-mf",
                             help="Path ou nome do modelo de ASR para transcrição fonética (não será usado para teste)")
    test_parser.add_argument("--device", "-d",
                             help="Device a ser passado como argumento para o framework de transcrição do Hugging Face."
                                  "-1 para usar a CPU.", 
                             default=0)
    test_parser.add_argument("--metrics",
                             nargs="+", 
                             help="Métricas de teste. Opções disponíveis: wer mer wil cer all",
                             default="all")
    test_parser.add_argument("--average-from-sentences",
                             help="O resultado das métricas será realizado a partir da média das sentenças de cada áudio."
                                  "Por padrão, todas as sentenças e as predições são usadas para calcular as métricas.",
                             action="store_true")
    csv_parser = parser.add_argument_group('Opções dos arquivos CSV de saída')
    csv_parser.add_argument("--ptbr",
                            help="Usa o separador de ponto-e-virgula (;) e o formato de número em PT-BR (XX,XX)", 
                            action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(levelname)s:%(filename)s:%(lineno)s:%(message)s", 
                        level=args.log_level, 
                        filename=args.log_file, 
                        filemode='w')

    prepare_output_dirs(args)
    check_files_list(args)
    check_args(args)

    main(args)
