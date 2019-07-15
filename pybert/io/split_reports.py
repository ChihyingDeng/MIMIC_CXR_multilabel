import pdb
import pandas as pd
import os
import re
import glob
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
#import nltk
#nltk.download('punkt')

class SplitReports(object):
    def __init__(self, raw_reports_dir, raw_data_path):
        self.raw_reports_dir = raw_reports_dir
        self.raw_data_path = raw_data_path

    # create a custom iterator that also returns the span of the sentence
    def _sentence_spans(self, text):
        tokens = sent_tokenize(text)
        offset = 0
        for token in tokens:
            offset = text.find(token, offset)
            yield token, offset, offset+len(token)
            offset += len(token)

    def _valid_section(self, name):
        if name in ['history', 'comparison', 'examination', 'indication', 'preamble',\
                    'idication', 'technique', 'addendum', 'clinical information']:
            return False
        # anything that passes these tests is OK
        return True

    def split(self):
        reports=[]
        default_path = os.getcwd()
        os.chdir(self.raw_reports_dir)

        file_list=[]
        for file in glob.glob("*.deid"):
            with open(file, 'r') as f:
                txt = ''.join(f.readlines())
            reports.append(txt.rstrip('\r'))
            file_list.append(file.split('.')[0])

        os.chdir(default_path)

        # ------------- Split the  reports into section -------------
        p_section = re.compile('\n ?([A-Z ()/,-]+):\s+(.+?)\n ?(\n|\Z)', re.DOTALL)

        sections_all = list()
        names_all = list()
        clipnums = list()
        no_sections = list()
        section_start_idx_all = list()

        for fn in range(len(reports)):
            clipnum = int(fn)
            rr_text = reports[fn]

            # do not process if no report is present
            if len(rr_text) == 0:
                continue
            sections = list()
            section_names = list()
            section_start_idx = list()
            idx = 0
            s = p_section.search(rr_text, idx)

            if s:
                sections.append(rr_text[0:s.start(1)])
                section_names.append('preamble')
                section_start_idx.append(0)
                while s:
                    current_section = s.group(1)
                    sections.append(s.group(2).replace('\n', ' ').strip('\n '))
                    section_names.append(current_section.lower())
                    section_start_idx.append(s.start(2))

                    idx = s.end(2)
                    s = p_section.search(rr_text, idx)

            # add the last remaining
            if idx != len(rr_text):
                sections.append(rr_text[idx:])
                section_names.append('remainder')
                section_start_idx.append(idx)
            else:
                no_sections.append(fn)
                sections.append(rr_text)
                section_names.append('full report')
                section_start_idx.append(0)

            sections_all.append(sections)
            names_all.append(section_names)
            section_start_idx_all.append(section_start_idx)
            clipnums.append(fn)


        # ------------- Split the sections into sentences -------------
        # create a simple list with no nesting:
        # clipnum, section_index, section, sentence number, start, stop, sentence
        sentences = list()
        for i, c in tqdm(enumerate(clipnums), total=len(clipnums)):
            section_start_idx = section_start_idx_all[i]
            id = 0
            for j, section in enumerate(sections_all[i]):
                start_add = section_start_idx[j]
                # only include sections pertaining to the x-ray
                if not self._valid_section(names_all[i][j]):
                    continue
                n = 0
                for sent, start, end in self._sentence_spans(section):
                    start += start_add
                    end += start_add
                    sentences.append([file_list[i]+'_'+str(id), sent, j, names_all[i][j], n,
                                      start, end])
                    n+=1
                    id+=1

        # convert this list to a dataframe
        df = pd.DataFrame(sentences,
                          columns=['idx', 'sentence', 'section_index', 'section',
                                   'sentence_index', 'start', 'stop'])
        df = df[['idx', 'sentence']]
        df.to_csv(self.raw_data_path, index=False)

        del sentences
