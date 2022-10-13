TEXTGRIDS="
    NURCSP_CM_v3/SP_D2_062.TextGrid
    NURCSP_CM_v3/SP_D2_255.TextGrid
    NURCSP_CM_v3/SP_D2_333.TextGrid
    NURCSP_CM_v3/SP_D2_343.TextGrid
    NURCSP_CM_v3/SP_D2_360.TextGrid
    NURCSP_CM_v3/SP_D2_396.TextGrid
    NURCSP_CM_v3/SP_DID_018.TextGrid
    NURCSP_CM_v3/SP_DID_137.TextGrid
    NURCSP_CM_v3/SP_DID_161.TextGrid
    NURCSP_CM_v3/SP_DID_208.TextGrid
    NURCSP_CM_v3/SP_DID_234.TextGrid
    NURCSP_CM_v3/SP_DID_235.TextGrid
    NURCSP_CM_v3/SP_DID_242.TextGrid
    NURCSP_CM_v3/SP_DID_250.TextGrid
    NURCSP_CM_v3/SP_DID_251.TextGrid
    NURCSP_CM_v3/SP_EF_124.TextGrid
    NURCSP_CM_v3/SP_EF_153.TextGrid
    NURCSP_CM_v3/SP_EF_156.TextGrid
    NURCSP_CM_v3/SP_EF_377.TextGrid
    NURCSP_CM_v3/SP_EF_388.TextGrid
    NURCSP_CM_v3/SP_EF_405.TextGrid
" 

AUDIO_FILES="
    NURCSP_CM_v3/SP_D2_062.wav
    NURCSP_CM_v3/SP_D2_255.wav
    NURCSP_CM_v3/SP_D2_333.wav
    NURCSP_CM_v3/SP_D2_343.wav
    NURCSP_CM_v3/SP_D2_360.wav
    NURCSP_CM_v3/SP_D2_396.wav
    NURCSP_CM_v3/SP_DID_018.wav
    NURCSP_CM_v3/SP_DID_137.wav
    NURCSP_CM_v3/SP_DID_161.wav
    NURCSP_CM_v3/SP_DID_208.wav
    NURCSP_CM_v3/SP_DID_234.wav
    NURCSP_CM_v3/SP_DID_235.wav
    NURCSP_CM_v3/SP_DID_242.wav
    NURCSP_CM_v3/SP_DID_250.wav
    NURCSP_CM_v3/SP_DID_251.wav
    NURCSP_CM_v3/SP_EF_124.wav
    NURCSP_CM_v3/SP_EF_153.wav
    NURCSP_CM_v3/SP_EF_156.wav
    NURCSP_CM_v3/SP_EF_377.wav
    NURCSP_CM_v3/SP_EF_388.wav
    NURCSP_CM_v3/SP_EF_405.wav
"

MODELS="
    Edresson/wav2vec2-large-xlsr-coraa-portuguese
    lgris/bp400-xlsr
    alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-plus-gain-normalization-plus-selective-data-augmentation
    jonatasgrosman/wav2vec2-xls-r-1b-portuguese" 

run() {
    for model in ${MODELS[@]}; do

        if [ ! -f "./output/accept_all/${model}/summary.csv" ]; then
            mkdir -p "./output/accept_all/${model}" 
            python3 test_asr.py \
                -t $TEXTGRIDS \
                -f $AUDIO_FILES \
                -o ./output/accept_all/${model}/ \
                --save-sentences-json-dir ./output/accept_all \
                --save-skipped-sentences-json-dir ./output/accept_all \
                --save-sentences-csv-dir ./output/accept_all \
                --save-skipped-sentences-csv-dir ./output/accept_all \
                --metrics wer mer wil cer \
                --average-from-sentences \
                -L INFO \
                --ptbr \
                --accept-all \
                -m "$model" \
                --generate-word-timestamps \
                --generate-char-timestamps \
                --log-file "./output/accept_all/${model}/output.log" > "./output/accept_all/${model}/output.txt"
        fi
        if [ ! -f "./output/ignore_all/${model}/summary.csv" ]; then
            mkdir -p "./output/ignore_all/${model}"
            python3 test_asr.py \
                -t $TEXTGRIDS \
                -f $AUDIO_FILES \
                -o ./output/ignore_all/${model}/ \
                --save-sentences-json-dir ./output/ignore_all \
                --save-skipped-sentences-json-dir ./output/ignore_all \
                --save-sentences-csv-dir ./output/ignore_all \
                --save-skipped-sentences-csv-dir ./output/ignore_all \
                --metrics wer mer wil cer \
                --average-from-sentences \
                -L INFO \
                --ptbr \
                --ignore-all \
                -m "$model" \
                --generate-word-timestamps \
                --generate-char-timestamps \
                --log-file "./output/ignore_all/${model}/output.log" > "./output/ignore_all/${model}/output.txt"
        fi
    done;

    for model in ${MODELS[@]}; do
        for ignop in "incomprehensible_sentences hypothesis_sentences" "sentences_with_annotation_parts" "overlap_sentences" "incomprehensible_sentences sentences_with_annotation_parts overlap_sentences"; do
            mkdir -p "./output/ignore_${ignop// /_}/${model}"
            if [ ! -f "./output/ignore_${ignop// /_}/${model}/summary.csv" ]; then
                python3 test_asr.py \
                    -t $TEXTGRIDS \
                    -f $AUDIO_FILES \
                    -o "./output/ignore_${ignop// /_}/${model}" \
                    --save-sentences-json-dir "./output/ignore_${ignop// /_}" \
                    --save-skipped-sentences-json-dir "./output/ignore_${ignop// /_}" \
                    --save-sentences-csv-dir "./output/ignore_${ignop// /_}" \
                    --save-skipped-sentences-csv-dir "./output/ignore_${ignop// /_}" \
                    --metrics wer mer wil cer \
                    --average-from-sentences \
                    -L INFO \
                    --ptbr \
                    --ignore-sentences-with $ignop \
                    -m "$model" \
                    --generate-word-timestamps \
                    --generate-char-timestamps \
                    --log-file "./output/ignore_${ignop// /_}/${model}/output.log" > "./output/ignore_${ignop// /_}/${model}/output.txt"
            fi
            done;
    done;
}

run
