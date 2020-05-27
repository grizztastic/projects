import gensim.models
import numpy as np
from sklearn.decomposition import PCA
import json
from tqdm import tqdm
#Reference: https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py

class WordEmbeddingDebiaser:

    def __init__(
            self,
            embedding_file_path,
            definitional_file_path='./data/definitional_pairs.json',
            equalize_file_path='./data/equalize_pairs.json',
            gender_specific_file_path='./data/gender_specific_full.json'
    ):

        self.model = gensim.models.KeyedVectors.load_word2vec_format(
            embedding_file_path, binary=True
        )

        # collect first 300000 words
        self.words = sorted([w for w in self.model.vocab],
                            key=lambda w: self.model.vocab[w].index)[:300000]

        # all vectors in an array (same order as self.words)
        self.vecs = np.array([self.model[w] for w in self.words])
        tqdm.write('vectors loaded')
        # should take 2-5 min depending on your machine

        self.n, self.d = self.vecs.shape

        # word to index dictionary
        self.w2i = {w: i for i, w in enumerate(self.words)}

        # Some relevant words sets required for debiasing
        with open(definitional_file_path, "r") as f:
            self.definition_pairs = json.load(f)

        with open(equalize_file_path, "r") as f:
            self.equalize_pairs = json.load(f)

        with open(gender_specific_file_path, "r") as f:
            self.gender_specific_words = json.load(f)
        self._normalize()

    # Some potentially helpful functions, you don't have to use/implement them.
    def _normalize(self):
        """
        normalize self.vecs
        """
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]

    def _drop(self, u, v):
        """
        remove a direction v from u
        """
        return u - v * u.dot(v) / v.dot(v)

    def w2v(self, word):
        """
        for a word, return its corresponding vector
        """
        return self.vecs[self.w2i[word]]

    def debias(self):
        self.gender_direction = self.identify_gender_subspace()
        self.neutralize()
        self.equalize()

    def identify_gender_subspace(self):
        """Using self.definitional_pairs to identify a gender axis (1 dimensional).

          Output: a gender direction using definitonal pairs

        ****Note****

         no other unimported packages listed above are allowed, please use
         numpy.linalg.svd for PCA

        """
        # SOLUTION
        # get indexes of 'female' and 'male' related words in definitional pairs
        index_list = [[self.words.index(f), self.words.index(m)] \
                      for f, m in self.definition_pairs]

        center_list = [[(self.vecs[pair[0]] + self.vecs[pair[1]]) / 2] \
                       for pair in index_list]  # get average the data pair

        differences = [[self.vecs[index_list[i][0]] - center_list[i], \
                        self.vecs[index_list[i][1]] - center_list[i]] \
                       for i in range(len(center_list))]  # calc dist away from mean

        differences = np.array([pair[i][0] for pair in differences for i in range(len(pair))])  # turn into array

        _, _, Vh = np.linalg.svd(differences, full_matrices=False)  # SVD to compute PCA on the data
        self.gend_direct = -Vh[0]  # gender direction denoted by 1st eigenvector
        '''Checked my self.pca against the code below for sklearn.decomposition.PCA method
        pca = PCA(n_components=1)
        pca = pca.fit(differences)
        pca_vals = pca.components_[0]
        print(self.gend_direct.round(2) == pca_vals.round(2)) #True
        '''

        # END OF SOLUTION

    def neutralize(self):
        """Performing the neutralizing step: projecting all gender neurtal words away
        from the gender direction

        No output, please adjust self.vecs

        """
        # SOLUTION
        # modify self.vecs only if the word is not in the gender specific words
        self.vecs = np.array([self._drop(self.vecs[i], self.gend_direct) \
                                  if self.words[i] not in self.gender_specific_words \
                                  else self.vecs[i] for i in range(len(self.words))])
        self._normalize()
        # END OF SOLUTION

    def equalize(self):
        """Performing the equalizing step: make sure all equalized pairs are
        equaldistant to the gender direction.

        No output, please adapt self.vecs

        """
        # SOLUTION
        dist = [(self.w2v(f) + self.w2v(m)) / 2 for f, m in self.equalize_pairs]  # dist for words in equalize pairs
        remove = [self._drop(dist[i], self.gend_direct) for i in range(len(dist))]  # removing directions

        # distance metric to know how far to move the words to make them equidistant in step below
        v = [np.sqrt(1 - np.linalg.norm(remove[i]) ** 2) if np.dot(dist[i] * 2, self.gend_direct) < 0 \
                 else -np.sqrt(1 - np.linalg.norm(remove[i]) ** 2) for i in range(len(remove))]

        idx = 0
        '''update self.vecs accordingly equalizing distance between words in 
           equalize pairs and gender neutral words for the gender subspace'''
        for f, m in self.equalize_pairs:
            self.vecs[self.words.index(f)] = v[idx] * self.gend_direct + remove[idx]
            self.vecs[self.words.index(m)] = -v[idx] * self.gend_direct + remove[idx]
            idx += 1
        self._normalize()
        # END OF SOLUTION

    def compute_analogy(self, w3, w1='woman', w2='man'):
        """input: w3, w1, w2, satifying the analogy w1: w2 :: w3 : w4

        output: w4(a word string) which is the solution to the analogy (w4 is
          constrained to be different from w1, w2 and w3)

        """
        diff = self.w2v(w2) - self.w2v(w1)
        vec = diff / np.linalg.norm(diff) + self.w2v(w3)
        vec = vec / np.linalg.norm(vec)
        if w3 == self.words[np.argsort(vec.dot(self.vecs.T))[-1]]:
            return self.words[np.argsort(vec.dot(self.vecs.T))[-2]]
        return self.words[np.argmax(vec.dot(self.vecs.T))]


if __name__ == '__main__':

    # Original Embedding

    we = WordEmbeddingDebiaser('./data/GoogleNews-vectors-negative300.bin')

    print('=' * 50)
    print('Original Embeddings')
    # she-he analogy evaluation
    w3s1 = [
        'her', 'herself', 'spokeswoman', 'daughter', 'mother', 'niece',
        'chairwoman', 'Mary', 'sister', 'actress'
    ]
    w3s2 = [
        'nurse', 'dancer', 'feminist', 'baking', 'volleyball', 'softball',
        'salon', 'blond', 'cute', 'beautiful'
    ]

    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    we.debias()

    print('=' * 50)
    print('Debiased  Embeddings')
    # she-he analogy evaluation
    w4s1 = [we.compute_analogy(w3) for w3 in w3s1]
    w4s2 = [we.compute_analogy(w3) for w3 in w3s2]

    print('Appropriate Analogies')
    for w3, w4 in zip(w3s1, w4s1):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))

    print('Potentially Biased Analogies')
    for w3, w4 in zip(w3s2, w4s2):
        print("'woman' is to '%s' as 'man' is to '%s'" % (w3, w4))