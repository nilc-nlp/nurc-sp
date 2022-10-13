# Bringing NURC/SP to Digital Life: the Role of Open-source Automatic Speech Recognition Model

Paper: [_coming soon_!]()

## Abstract

The NURC Project that started in 1969 to study the cultured linguistic urban norm spoken in five Brazilian capitals, was responsible for compiling a large corpus for each capital. The digitized NURC/SP comprises 375 inquiries in 334 hours of recordings taken in Sao Paulo capital. Although 47 inquiries have transcripts, there was no alignment between the audio-transcription, and 328 inquiries were not transcribed. This article presents an evaluation and error analysis of three automatic speech recognition models trained with spontaneous speech in Portuguese and one model trained with prepared speech. The eval- uation allowed us to choose the best model, in terms of WER and CER, in a manually aligned sample of NURC/SP, to automatically transcribe 284 hours

## Evaluated models

We evaluated the following models:

- [Edresson/wav2vec2-large-xlsr-coraa-portuguese](https://huggingface.co/Edresson/wav2vec2-large-xlsr-coraa-portuguese)
- [alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization-sna](https://huggingface.co/alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization-sna)
- [jonatasgrosman/wav2vec2-xls-r-1b-portuguese](https://huggingface.co/jonatasgrosman/wav2vec2-xls-r-1b-portuguese)
- [lgris/bp400-xlsr](https://huggingface.co/lgris/bp400-xlsr)

## Running tests

The test can be executed using the main script test_asr.py. For example, to ignore incomprehensible sentences of the NURC/SP annotations:

```sh
python test_asr.py \
    -t $TEXTGRIDS \
    -f $AUDIO_FILES \
    -m $MODEL \
    --ignore-sentences-with incomprehensible_sentences \
    -o "./output/ignore_incomprehensible_sentences/${model}" \
    --save-sentences-json-dir "./output/ignore_incomprehensible_sentences" \
    --save-skipped-sentences-json-dir "./output/ignore_incomprehensible_sentences" \
    --save-sentences-csv-dir "./output/ignore_incomprehensible_sentences" \
    --save-skipped-sentences-csv-dir "./output/ignore_incomprehensible_sentences" \
    --metrics wer mer wil cer \
    --average-from-sentences \
    --ptbr \
    --generate-word-timestamps \
    --generate-char-timestamps 
```

See `python test_asr.py --help` for details.

### Download dataset

The dataset used in these research (NURC/SP-MC) can be obtained at the (oficial corpus website)[https://portulanclarin.net/repository/browse/391c9bf232cd11ed84e202420a87010e52130324c1fe4a2981c00cbce6261766/].

## Results

We evaluated the MC using the models listed above and the best model in several scenarios. The detailed analysis is described in the paper.

Average results for all inquiries of MC:

![results](https://user-images.githubusercontent.com/34692520/195699302-2f6bcb82-4684-4db7-acdd-548c9f6ab9e6.png)

__Note:__ The phonetic transcription is preliminary and might not be accurate, specially its alignments.

## Citation

_coming soon_