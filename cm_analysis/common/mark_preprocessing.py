import re
import string
import logging

from num2words import num2words


class MarkPreprocessing:

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõũúû1234567890%\-\n/\\ "

    exp_let = ['a', 'bê', 'cê', 'dê', 'e', 'éfe', 'gê', 'agá', 'i', 'jota', 'cá', 
            'éle', 'eme', 'ene', 'o', 'pê', 'quê', 'érre',
            'ésse', 'tê', 'u', 'vê', 'dáblio', 'xis', 'ípsilom', 'zê', 'c cedilha']

    siglas = ["abi", "abnt", "agu", "avc", "bc", "bn", "bndes", "bnh", "br", "cbf", 
            "cd", "cgu", "cjf", "clt", "cn", "cnj", "cnmp", "cnp", "cnpq", "cpf", 
            "cpfl", "csjt", "ddd", "df", "dna", "cpu", "ect", "fgts", "fmi", "gsi", 
            "gps", "http", "ibge", "igpm", "iml", "inps", "inss", "iof", "ipi", 
            "iptu", "irpf", "isv", "lc", "ldb", "mc", "mct", "md", "mds", "mdic", 
            "mds", "mf", "mj", "mdc", "mma", "mmc", "mme", "mpa", "mpdft", "mpf", 
            "mpm", "mpog", "mps", "mpt", "mpu", "mre", "ms", "mt", "mte", "oab", 
            "pcb", "pco", "pdt", "phs", "pmdb", "pmn", "pp", "ppl", "pps", "prb", 
            "prp", "prtb", "psb", "psc", "psd", "ps", "psdb", "psdc", "psl", "pstu", 
            "pt", "ptb", "ptc", "ptn", "pv", "rg", "rgps", "rj", "rn", "sf", "sg", 
            "sms", "sp", "spc", "sri", "stf", "stj", "stm", "tse", "tst", "ufrj", 
            "ufmg", "ufpr", "ufc", "vpr", "www"]

    def __init__(
        self, 
        ignore_abreviations=True,
        ignore_sentences_with_annotation_parts=True,
        ignore_incomprehensible_sentences=False,
        ignore_overlap_sentences=False,
        ignore_hypothesis_sentences=False,
        remove_incomprehensible_parts=True,
        normalize_text=True,
        _ignore_empty_sentences=True,   # Must be True
        _remove_annotation_parts=True,  # Must be True
        _remove_extra_characters=True,  # Must be True
        _remove_extra_spaces=True       # Must be True
    ):
        self._ignore_abreviations = ignore_abreviations
        self._ignore_empty_sentences = _ignore_empty_sentences
        self._ignore_incomprehensible_sentences = ignore_incomprehensible_sentences
        self._ignore_hypothesis_sentences = ignore_hypothesis_sentences
        self._ignore_overlap_sentences = ignore_overlap_sentences
        self._ignore_sentences_with_annotation_parts = ignore_sentences_with_annotation_parts
        self._remove_annotation_parts = _remove_annotation_parts
        self._remove_extra_characters = _remove_extra_characters
        self._remove_extra_spaces = _remove_extra_spaces
        self._remove_incomprehensible_parts = remove_incomprehensible_parts
        self._normalize_text = normalize_text

    def contains_num(s):
        return any(i.isdigit() for i in s)

    def normalize(text):
        # éh, eh
        filled_pause_eh = ["éh", "ehm", "ehn", "he", "éhm", "éhn", "hé"]
        # uh, hum, hm, uhm
        filled_pause_uh = ["hum", "hm", "uhm", "hu", "uhn"]
        # uhum, aham
        filled_pause_aham = ["uhum", "uhun", "unhun", "unhum", "umhun",
                            "umhum", "hunhun", "humhum", "hanhan", "ahan", "uhuhum"]
        # ah, hã, ãh, ã
        filled_pause_ah = ["hã", "ãh", "ã", "ah", "ahn", "han", "ham"]
        if text == "$$$":
            return '###'
        elif text == "@@@":
            return '###'

        # Remove marcas de truncamento (\)
        elif '/' in text:
            text = '###'
            return text
        # Remove string vazia
        if(len(text) == 0):
            text = "###"
            return text
        # Remove qualquer caractere fora do alfabeto
        text = re.sub("[^{}]".format(MarkPreprocessing.alphabet), '', text)
        # Converte maiúsculas para minúsculas
        text = text.lower()

        # Remove hífens
        text = re.sub('\-+', " ", text)

        # Remove espaços múltiplos
        text = re.sub(' +', ' ', text)

        # Remove espaços no começo e no final da string
        text = text.strip()

        # Separa o texto em palavras e itera por elas
        words = text.split(' ')
        new_words = []
        for word in words:
            if word == '' or word == ' ':
                continue

            if word == "hhh":
                continue

            # Substitui ehhhhhh por eh e afins
            word = re.sub("h+", "h", word)

            if word in filled_pause_eh:
                word = "eh"

            elif word in filled_pause_uh:
                word = "uh"

            elif word in filled_pause_aham:
                word = "aham"

            elif word in filled_pause_ah:
                word = "ah"

            # Expande siglas
            if word in MarkPreprocessing.siglas:
                sigla_exp = ""
                for l in word:
                    if l == 'ç':
                        sigla_exp = sigla_exp + MarkPreprocessing.exp_let[-1]
                    else:
                        sigla_exp = sigla_exp + MarkPreprocessing.exp_let[ord(l) - 97]
                word = sigla_exp

            # Substitui 33% por 33 por cento e afins
            word = re.sub("\d+[%]", lambda x: x.group()+" por cento", word)
            word = re.sub("%", "", word)

            # Trata casos como 5o
            word = re.sub("\d+[o]{1}", lambda x: num2words((x.group()
                                                            [:-1]), to='ordinal', lang='pt_BR'), word)
            # Trata casos como 5a, convertendo para o ordinal masculino primeiramente
            ref = word
            word = re.sub("\d+[a]{1}", lambda x: num2words((x.group()
                                                            [:-1]), to='ordinal', lang='pt_BR'), word)
            # Se ocorreu match com ordinal feminino e foi convertido para o masculino, separamos a nova frase e trocamos os 'o's finais por 'a's
            if word != ref:
                segs = word.split(' ')
                word = ''
                for seg in segs:
                    word = word + seg[:-1] + 'a' + ' '
                # Elimina o espaço adicional da última iteração
                word = word[:-1]

            if MarkPreprocessing.contains_num(word):
                segs = word.split(' ')
                word = ''
                for seg in segs:
                    if seg.isnumeric():
                        seg = num2words(seg, lang='pt_BR')
                    word = word + seg + ' '
                # Elimina o espaço adicional da última iteração
                word = word[:-1]
            new_words.append(word)

        # Reconstrói texto com palavras alteradas
        text = ''
        for new_word in new_words:
            # print(new_word)
            text = text + new_word + ' '

        # Elimina o espaço adicional da última iteração
        text = text[:-1]

        return text

    letters_list = string.ascii_lowercase
    abreviacao_list = [ str(x.upper())+'. ' for x in letters_list]+['yyyy']+['xxxx']  

    def detect_abreviations(texto):
        for a in MarkPreprocessing.abreviacao_list:
            if( texto.find(a)!= -1):
                return True
        return False    


    def __call__(self, mark):
        """
        Marking pre-processing function.

        Returns preprocessed or null (skipped) text based on the
        object settings.

        None == Ignored
        """
        logging.debug(f"Processando marcação \"{mark}\"")
        mark = mark.replace('Doc.', '')
        mark = mark.replace('L1', '')
        mark = mark.replace('L2', '')
        mark = mark.replace('Inf', '')

        if self._ignore_incomprehensible_sentences:
            # Ignora trechos com (frase incompreensível)
            if '( )' in mark:
                logging.debug(f"\tIncomprehensible snippets detected. Ignoring \"{mark}\"")
                return None

        if self._ignore_hypothesis_sentences:
            # Ignora trechos com (frase incompreensível)
            if re.search('\(.*\)', mark):
                logging.debug(f"\tExcerpts with detected hypotheses. Ignoring \"{mark}\"")
                return None

        if self._ignore_overlap_sentences:
            # Ignora trechos com [fala com sobreposição...
            if '[' in mark or ']' in mark:
                logging.debug(f"\tOverlapping snippets detected. Ignoring \"{mark}\"")
                return None

        if self._ignore_sentences_with_annotation_parts:
            # Ignora trechos com ((anotação))
            if re.search('\(\(.*\)\)', mark):
                logging.debug(f"\tAnnotated snippets detected. Ignoring \"{mark}\"")
                return None
        
        if self._ignore_abreviations:
            # Ignora abreviações
            if MarkPreprocessing.detect_abreviations(mark):
                logging.debug(f"\tAbbreviation detected. Ignoring \"{mark}\"")
                return None

        if self._remove_incomprehensible_parts:
            # Remove tudo entre (( ))
            text = re.sub('\(.*\)', '', mark)
            logging.debug(f"\tRemoval of incomprehensible parts: \"{text}\"")

        if self._remove_annotation_parts:
            # Remove tudo entre (( ))
            text = re.sub('\(\(.*\)\)', '', mark)
            logging.debug(f"\tRemoving annotations: \"{text}\"")
        
        if self._remove_extra_characters:
            # Remove caracteres extras
            text = re.sub('[\,\?\.\!\;\:\"\(\)\[\]/-]', '', text)
            logging.debug(f"\tRemoval of extra characters: \"{text}\"")
        
        if self._remove_extra_spaces:
            # Remove espaços extras
            text = ' '.join(text.split())
            logging.debug(f"\tRemoving extra spaces: \"{text}\"")

        logging.debug(f"\tPre-processed text: \"{text}\"")

        if self._normalize_text:
            text = MarkPreprocessing.normalize(text)
            logging.debug(f"\tNormalized text: \"{text}\"")

        if text == '###' or text == ' ' or text == '':  # Vazio
            if self._ignore_empty_sentences:
                logging.debug(f"\tEmpty text detected: \"{text}\"")
                return None
            else:
                text = ' '

        logging.debug(f"\t\"{mark}\" -> \"{text}\"")

        return text
