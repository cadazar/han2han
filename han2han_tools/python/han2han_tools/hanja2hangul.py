# coding:utf-8
# 원작:마소리스(masoris@gmail.com)'
##################################
# 갱신:jek.cl.nlp@gmail.com
# dic_c2h, ncovert, tqdm, outfn
# 두음법칙
#https://ko.wikipedia.org/wiki/%EB%91%90%EC%9D%8C_%EB%B2%95%EC%B9%99
##################################
# Further edited: cadams@sogang.ac.kr for use with Han2Han-Embeddings

import re
import sys
try:
    from importlib import resources
except ImportError:
    # Python < 3.7
    import importlib_resources as resources

# 유니코드 한자 범위: r00 ~ r10
r00 = "\U00004e00-\U00009fff" # (한중일 통합 한자)
r01 = "\U00003400-\U00004db5" # (한중일 통합 한자 확장 A)
r02 = "\U00020000-\U0002a6d6" # (한중일 통합 한자 확장 B)
r03 = "\U0002a700-\U0002b734" # (한중일 통합 한자 확장 C)
r04 = "\U0002b740-\U0002b81f" # (한중일 통합 한자 확장 D)
r05 = "\U0002b820-\U0002ceaf" # (한중일 통합 한자 확장 E)
r06 = "\U0002ceb0-\U0002ebe0" # (한중일 통합 한자 확장 F)
r07 = "\U0000f900-\U0000faff" # (한중일 호환용 한자)
r08 = "\U0002f800-\U0002fa1f" # (한중일 호환용 한자 보충)
r09 = "\U000031c0-\U000031ef" # (한중일 한자 획)
r10 = "\U00002ff0-\U00002fff" # (한자 생김꼴 지시 부호)

dooeums0 = '녀, 념, 뇨, 뉴, 니, 랴, 려, 례, 료, 류, 리, 림, 라, 래, 로, 뢰, 루, 르'.split(', ')
dooeums1 = '여, 염, 요, 유, 이, 야, 여, 예, 요, 유, 이, 임, 나, 내, 노, 뇌, 누, 느'.split(', ')
dooeums = {k:v for k, v in zip(dooeums0, dooeums1)}

kor_begin     = 44032
kor_end       = 55203
chosung_base  = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 
        'ㅡ', 'ㅢ', 'ㅣ']

jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']


def compose(chosung, jungsung, jongsung):
    return chr(kor_begin + chosung_base * chosung_list.index(chosung) + jungsung_base * jungsung_list.index(jungsung) + jongsung_list.index(jongsung))


def decompose(c):
    if not character_is_korean(c):
        return None
    i = to_base(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')    
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base 
    jong = ( i - cho * chosung_base - jung * jungsung_base )    
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])


def character_is_korean(c):
    i = to_base(c)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)


def to_base(c):
    if sys.version_info.major == 2:
        if type(c) == str or type(c) == unicode:
            return ord(c)
        else:
            raise TypeError
    else:
        if type(c) == str or type(c) == int:
            return ord(c)
        else:
            raise TypeError


class Hanja2Hangul:
    def __init__(self, mode=2, uniconv=True):
        self.mode = mode
        self.uniconv = uniconv
        self.dic_c2h = self.populate_dicts(mode)
        self.allhanja = fr'{r00}{r01}{r02}{r03}{r04}{r05}{r06}{r07}{r08}{r09}{r10}'
        self.ishanja = re.compile(r'[{}]'.format(self.allhanja))
        self.doublespace_pattern = re.compile(r'\s+') # 두칸 이상의 공백을 하나로 치환
        self.kojpbrackets = r'가-힇ㄱ-ㅎㅏ-ㅣ·ぁ-ゔァ-ヴー々〆〤【】『』「」\(\)'
        self.cjk = re.compile(r'[^{} {}\d]+'.format(self.allhanja, self.kojpbrackets))

    def hanhansplit(self, line):
        splittingregex = re.compile(r'[{}]+|[^{}]+'.format(self.allhanja, self.allhanja))
        toreturn = []
        for splitstr in splittingregex.findall(line):
            newsplit = splitstr.split()
            toreturn.extend(newsplit)
        return toreturn 
        
    def convert(self, line, uniconv:bool=True): #Convert Hanja to Hangul
        origline = line
        
        # always check for multi-character matches first to get contemporary spellings
        # this prevents 音樂 from becoming 음락 when it should be 음악
        converted_parts = []
        i = 0
        while i < len(line):
            matched = False
            # try to match longest possible substring starting at position i
            for j in range(min(i + 10, len(line)), i, -1):  # limit max length to 10 chars for efficiency
                if line[i:j] in self.dic_c2h:
                    converted_parts.append(self.dic_c2h[line[i:j]])
                    i = j
                    matched = True
                    break
            
            if not matched:
                # no multi-char match found, try single char
                char = line[i]
                nchar = self.dic_c2h.get(char, '')
                if nchar == '':
                    converted_parts.append(char)
                else:
                    # apply 두음법칙 only for single character conversions
                    dnchar = dooeums.get(nchar, nchar)
                    converted_parts.append(dnchar)
                i += 1
        
        return ''.join(converted_parts)

    def hanjain(self, line): 
        return [bool(self.ishanja.match(c)) for c in line]

    def jamo_sentence(self, sent):
        def transform(char):
            if char == ' ':
                return char
            cjj = decompose(char)
            if len(cjj) == 1:
                return cjj
            cjj_ = ''.join(c if c != ' ' else '-' for c in cjj)
            return cjj_

        sent_ = []
        for char in sent:
            if character_is_korean(char):
                sent_.append(transform(char))
            else:
                sent_.append(char)
        sent_ = self.doublespace_pattern.sub(' ', ''.join(sent_))
        return sent_

    def jamo_to_word(self, jamo):
        jamo_list, idx = [], 0
        while idx < len(jamo):
            if not character_is_korean(jamo[idx]):
                jamo_list.append(jamo[idx])
                idx += 1
            else:
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3
        word = ""
        for jamo_char in jamo_list:
            if len(jamo_char) == 1:
                word += jamo_char
            elif jamo_char[2] == "-":
                word += compose(jamo_char[0], jamo_char[1], " ")
            else:
                word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
        return word

    def readdic(self, dic0, dic1, dic_c2h, dicfilename='./dict/dic1.txt'): #Add Dictionary to dic0, dic1

        # Use importlib.resources for loading package data
        try:
            # Python 3.9+
            files = resources.files('han2han_tools')
            dicfile = str(files / dicfilename)
        except AttributeError:
            # Python 3.7-3.8
            with resources.path('han2han_tools', dicfilename) as p:
                dicfile = str(p)
        dic = open(dicfile, 'r', encoding='utf-8')

        while(True):

            line = dic.readline()
            if len(line) == 0:
                break
            if line.find('\t') == -1:
                continue

            line = line.replace('\n','')
            if line[0] == '#':
                continue

            splited = line.rsplit('\t')
            if len(splited) <= 1:
                continue
            if len(splited[0]) == 0 or len(splited[1]) == 0:
                continue

            assert len(splited[0]) == len(splited[1]), 'Error: %s' % line

            dic0.append(splited[0])
            dic1.append(splited[1])
            dic_c2h[splited[0]] = splited[1]

        dic.close()

        return dic0, dic1, dic_c2h

    def populate_dicts(self, mode=1): 
        if(mode == 1): #unicode
            dic0 = []
            dic1 = []
            dic_c2h = {}
            dic0, dic1, dic_c2h = self.readdic(dic0, dic1, dic_c2h, './dict/dic0.txt')
            return dic_c2h

        elif(mode == 2): #hanja to hangul
            dic0 = []
            dic1 = []
            dic_c2h = {}
            dic0, dic1, dic_c2h = self.readdic(dic0, dic1, dic_c2h, './dict/dic0.txt')
            dic0, dic1, dic_c2h = self.readdic(dic0, dic1, dic_c2h, './dict/dic4.txt')
            dic0, dic1, dic_c2h = self.readdic(dic0, dic1, dic_c2h, './dict/dic1.txt')
            return dic_c2h

    def hprint(self, sentence):
        if any(self.hanjain(sentence)):
            print(f"{sentence}({self.convert(sentence)})")
        else:
            print(sentence)

    def h_out(self, sentence):
        if any(self.hanjain(sentence)):
            return f"{sentence}({self.convert(sentence)})"
        else:
            return sentence