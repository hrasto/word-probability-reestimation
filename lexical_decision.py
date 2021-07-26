import pandas as pd
import os
import numpy as np
from enum import Enum, IntEnum

class Form(Enum):
    raw = 1
    norm = 2
    lognorm = 3

class Group(IntEnum):
    first = 1
    second = 2

inaccurate_response=666

def z_lognorm(x: pd.Series):
    """ Standardization assuming a log-normal distribution """
    x = x - np.min(x) + 1
    base = x.median()
    x_norm = np.log(x)/np.log(base)-1
    std = x_norm.std()
    x_norm /= -std
    x_norm.loc[x_norm.isna()] = inaccurate_response
    return x_norm

def z_norm(x: pd.Series):
    """ Standardization assuming a normal distribution """
    mean = x.mean()
    std = x.std()
    x_norm = (x-mean)/std
    x_norm.loc[x_norm.isna()] = inaccurate_response
    return x_norm

class BLP(object):
    """Class managing the contents of the British Lexicon Project."""

    def __init__(self, path, trials='blp-trials.txt', stimuli='blp-stimuli.txt', nonwords=False, words: set={}, nrows=None):
        """Constructor

        Args:
            path (str): Path to the folder with extracted BLP files.
            trials (str, optional): Name of the text file with BLP trials. Defaults to 'blp-trials.txt'.
            stimuli (str, optional): Name of the text file with BLP stimuli. Defaults to 'blp-stimuli.txt'.
            nonwords (bool, optional): Whether or not to include non-words. Defaults to False.
            words (set, optional): Set of words to include. When empty, no filtering happens. Defaults to {}.
            nrows (int, optional): Limit the number of rows to read from BLP trials. Defaults to None. 
        """
        # load data
        trials = pd.read_csv(os.path.join(path, trials), sep='\t', nrows=nrows)
        trials = trials[trials['lexicality']=='W']
        # standardize rts
        trials['rt.norm'] = trials.groupby(['block', 'participant'])['rt'].transform(z_norm)
        trials['rt.lognorm'] = trials.groupby(['block', 'participant'])['rt'].transform(z_lognorm)
        trials = trials[['rt.norm', 'rt.lognorm', 'rt.raw', 'spelling', 'participant']]
        # restructure
        trials=pd.pivot(trials, 
                        index='spelling', 
                        columns='participant', 
                        values=['rt.raw', 'rt.norm', 'rt.lognorm'])
        # remove nan
        self.trials = trials.iloc[1:]
        # filter words
        if len(words) > 0:
            self.trials = self.trials.loc[words.intersection(self.trials.index)]
        # organize
        self.trials = self.trials.sort_index()
        self.rt_raw = self.trials['rt.raw']
        self.rt_norm = self.trials['rt.norm']
        self.rt_lognorm = self.trials['rt.lognorm']
        # identify groups
        self.words_idx_1 = self.rt_raw[1].isna()
        self.words_idx_2 = self.rt_raw[2].isna()
        self.cols_gr_1 = self.rt_raw[self.words_idx_2].iloc[0].isna()
        self.cols_gr_2 = ~self.cols_gr_1
        # use nans for inaccurate trials again
        for form, pt in self.trials.columns:
            to_replace = self.trials[form, pt]==inaccurate_response
            self.trials[form, pt][to_replace] = np.nan
        self.stimuli = pd.read_csv(os.path.join(path, stimuli), sep='\t', index_col='spelling')
        self.stimuli = self.stimuli.loc[self.trials.index]

    def get_words(self):
        """ Returns a pd.Series object containing words in loaded trials. """
        return self.trials.index

    def get_words_group(self, gr: Group = Group.first):
        """ Returns a pd.Series object containing words in given group. """
        return self.trials.index[self.words_idx_1 if gr==Group.first else self.words_idx_2]

    def get_rt_by_form(self, form: Form = Form.lognorm):
        """ Returns a pd.Dataframe with reaction times of given form. """
        if form==Form.raw:
            df = self.rt_raw  
        elif form==Form.norm:
            df = self.rt_norm
        elif form==Form.lognorm:
            df = self.rt_lognorm

        return df

    def get_participants(self, gr: Group = Group.first):
        """ Returns a pd.Series object with participants in given group. """
        participants = None
        # self.cols_gr_1/2 are pd.Series objects containing boolean values and indexed by words 
        if gr == Group.first:
            participants = self.cols_gr_1.index[self.cols_gr_1]
        elif gr == Group.second:
            participants = self.cols_gr_2.index[self.cols_gr_2]
        
        return participants
    
    def get_words_idx_group(self, gr: Group):
        if gr == 1:
            return self.words_idx_1
        elif gr == 2: 
            return self.words_idx_2

    def get_rt_group(self, gr: Group = Group.first, form: Form = Form.lognorm, mean=False):
        """Returns a dataframe with reaction times of the shape (nwords, nparticipants) for given group. 

        Args:
            gr (int, optional): Group; 1 or 2. Defaults to 1.
            form (Form, optional): Processing of the RTs; either 'Form.raw', 'Form.norm', or 'Form.lognorm'. Defaults to 'Form.raw'.
            mean (bool, optional): If true, computes mean across participants (ignores nans).

        Returns:
            pd.DataFrame: A dataframe with words as rows (index) and participants as columns
        """
        cols, df = None, None
        if gr == Group.first:
            #word_idx = self.words_idx_1
            cols = self.cols_gr_1.index[self.cols_gr_1]
        elif gr == Group.second:
            #word_idx = self.words_idx_2
            cols = self.cols_gr_2.index[self.cols_gr_2]
        df = self.get_rt_by_form(form)
        res = df.loc[:, cols]
        if mean:
            res = res.mean(axis=1)
        
        return res

    def get_rt_participant(self, pid, form: Form = Form.lognorm):
        """ Returns a pd.Series object with a given participant's reaction times in given form. """
        is_gr1 = pid in self.get_participants(Group.first)
        if not is_gr1:
            if pid not in self.get_participants(Group.second):
                raise ValueError('Participant {} does not exist.'.format(pid))
        #word_idx = self.words_idx_1 if is_gr1 else self.words_idx_2
        df = self.get_rt_by_form(form)

        return df.loc[:, pid]
    
    def get_stimuli(self, gr: Group = Group.first):
        """Returns a dataframe containing stimuli information for given group. 

        Args:
            gr (int, optional): Group; 1 or 2. Defaults to 1.

        Returns:
            pd.DataFrame: A dataframe with words as rows (index) and word information as columns.
        """
        word_idx = self.words_idx_1 if gr == Group.first else self.words_idx_2
        df = self.stimuli

        return df.loc[word_idx, :]