import os
import re
import logging
import json

import textgrid
import pandas as pd
from tabulate import tabulate
 
from common.mark_preprocessing import MarkPreprocessing


def parse_textgrids(args, save_textgrids=False):
    if args.accept_all or args.ignore_all:
        pre_process_nurcsp = MarkPreprocessing(
            ignore_abreviations=False if args.accept_all else True,
            ignore_incomprehensible_sentences=False if args.accept_all else True,
            ignore_sentences_with_annotation_parts=False if args.accept_all else True,
            ignore_overlap_sentences=False if args.accept_all else True,
            remove_incomprehensible_parts=False if args.accept_all else True
        )
    else:
        pre_process_nurcsp = MarkPreprocessing(
            ignore_incomprehensible_sentences="incomprehensible_sentences" in args.ignore_sentences_with,
            ignore_sentences_with_annotation_parts="sentences_with_annotation_parts" in args.ignore_sentences_with,
            ignore_overlap_sentences="overlap_sentences" in args.ignore_sentences_with
        )

    sentences = {}
    skipped_sentences = {}
    new_textgrids = {}

    for tf in args.textgrids:
        sample = os.path.splitext(os.path.basename(tf))[0]

        logging.debug(f"Reading TextGrid file: {tf}")

        tg = textgrid.TextGrid.fromFile(tf)
        new_tg = textgrid.TextGrid.fromFile(tf)
        
        sentences[sample] = []
        skipped_sentences[sample] = []

        for tier in tg:

            new_tier = textgrid.IntervalTier(
                name="N-"+tier.name,
                minTime=tier.minTime,
                maxTime=tier.maxTime
            )
            
            for it in tier:
                start_sec = it.minTime
                end_sec = it.maxTime
                mark = it.mark
                text = pre_process_nurcsp(mark)

                if text is not None:
                    sentences[sample].append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "mark": mark,
                        "text": text,
                        "duration": end_sec-start_sec
                    })
                else:  # Ignorado
                    text = ""
                    skipped_sentences[sample].append({
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "mark": mark,
                        "duration": end_sec-start_sec
                    })
                
                new_tier.add(minTime=start_sec, maxTime=end_sec, mark=text)
            
            if not "NTB" in tier.name:
                new_tg.tiers.append(new_tier)

        logging.info(f"Total sentences in the file {tf}: {len(sentences[sample])}")
        logging.info(f"Total skipped sentences from the file {tf}: {len(skipped_sentences[sample])}")

        new_textgrids[sample] = new_tg
        if save_textgrids:
            new_tg.write(os.path.join(args.out_dir, sample + '.TextGrid'))

    if args.save_sentences_json_dir is not None:
        logging.debug("Exporting JSON file sentences.json")
        with open(
            os.path.join(args.save_sentences_json_dir, "sentences.json"), 'w', encoding='utf8'
        ) as fp:
            json.dump(sentences, fp, ensure_ascii=False, indent=4)
        
    if args.save_skipped_sentences_json_dir is not None:
        logging.debug("Exporting JSON file skipped_sentences.json")
        with open(
            os.path.join(args.save_skipped_sentences_json_dir, "skipped_sentences.json"), 'w', encoding='utf8'
        ) as fp:
            json.dump(skipped_sentences, fp, ensure_ascii=False, indent=4)

    if args.save_sentences_csv_dir is not None and len(sentences[sample]) > 0:
        for tf in args.textgrids:
            sample = os.path.splitext(os.path.basename(tf))[0]
            logging.debug(f"Exporting CSV file of sentences from {sample}")

            sentences_pd = pd.DataFrame(sentences[sample])
            sentences_pd.to_csv(
                os.path.join(args.save_sentences_csv_dir, f"sentences_{sample}.csv"),
                sep=';' if args.ptbr else ',',
                decimal=',' if args.ptbr else None,
                index=False
            )
            if args.log_level in ('INFO', 'DEBUG'):
                print("Total sentences:")
                print(tabulate(sentences_pd, headers='keys', tablefmt='psql', maxcolwidths=45))
                print(tabulate(sentences_pd.describe(), headers='keys', tablefmt='psql', maxcolwidths=45))

    if args.save_skipped_sentences_csv_dir is not None and len(skipped_sentences[sample]) > 0:
        for tf in args.textgrids:
            sample = os.path.splitext(os.path.basename(tf))[0]
            logging.debug(f"Exporting CSV file of skipped sentences from {sample}")

            skipped_sentences_pd = pd.DataFrame(skipped_sentences[sample])
            skipped_sentences_pd.to_csv(
                os.path.join(args.save_skipped_sentences_csv_dir, f"skipped_sentences_{sample}.csv"),
                sep=';' if args.ptbr else ',',
                decimal=',' if args.ptbr else None,
                index=False
            )
            if args.log_level in ('INFO', 'DEBUG'):
                print("Ignored sentences:")
                print(tabulate(skipped_sentences_pd, headers='keys', tablefmt='psql', maxcolwidths=45))
                print(tabulate(skipped_sentences_pd.describe(), headers='keys', tablefmt='psql', maxcolwidths=45))

    return sentences, new_textgrids
    