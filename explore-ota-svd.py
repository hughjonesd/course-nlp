

# Exploring how to run basic NLP on "A Corpus of English Dialogues 1560-1760"

# import some stuff

import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# I unzipped my files and looked at one of them (first lines below):
# - lots of typography to indicate stage directions
# - lots of old-fashioned spelling, and use of single quotes for
#   abbreviation ("subdu'd", "resign'd")
# Maybe some stemming will help...
# But just using words and throwing away typography might be enough for now

# Filenames start with D1, D2, D3, D4, D5 and this indicates the period (40 year bins
# starting at 1560)
# The next 1 or 2 letters indicate the type of work:
# T - trials
# W - witness depositions
# C - drama comedy
# HO - didactic works, not language teaching (e.g. Flower of Friendshippe, Treason Made Manifest)
# HF, HE, HG - language teaching (F french, E english?, G German)
# F - prose fiction (e.g. Westward for Smelts, NB MUST READ THIS)
# M - miscellaneous


# Each file starts with 9 reference codes in angle brackets.
# angle brackets also used for <P xxx> page numbers

# then the following codes:
# (^..........^) (\..........\) [}..........}] [{..........{] [\..........\] [^..........^] [$..........$] [^...^] [^---^]
# font other than the basic font
# foreign language
# heading
# editorial emendation (i.e. a correction)
# editorial comment
# corpus compilers’ comment
# running text other than direct speech
# text on the line omitted
# text in the same sentence omitted

# We probably want to get rid of all of these except the first (different font)
# and maybe the heading and/or running text

# Potential problems: lots of names which will make for unique text identifiers,
# hence not very interesting clusters...!

# ## Pseudocode
# * Get the data into a form recognized by sklearn

data_folder = Path("ota_20.500.12024_2507/CEDPlain")

ced_filenames = os.listdir(data_folder)
ced_filenames = [data_folder/fn for fn in ced_filenames]

# we may want to use the "preprocessor" argument to get rid of some of the stuff above
# it takes a callable.
vectorizer = CountVectorizer(
                input      = 'filename',
                stop_words = 'english',
                encoding   = 'iso-8859-1',
             )

# from https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# If ‘filename’, the sequence passed as an argument to fit is expected to be a list
# of filenames that need reading to fetch the raw content to analyze.

# we use todense() here so that linalg below can get to work
# for speed, could use scipy.sparse.linalg
count_array = vectorizer.fit_transform(raw_documents = ced_filenames).todense()
vocab = vectorizer.get_feature_names()
# lots of english words, not very lemmatized
vocab[1000:1020]


U, s, Vh = linalg.svd(count_array, full_matrices=False)

# plotting the importance of the topics:
plt.plot(s)
plt.show()

# a little function to show the top topic words
num_top_words=8
def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]