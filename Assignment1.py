import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import string
from collections import defaultdict
import math

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the file paths for train.txt and val.txt. Files have to be in the same directory as the script
train_data = os.path.join(current_directory, "train.txt")
val_data = os.path.join(current_directory, "val.txt")

def preprocess(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    tokens = word_tokenize(data)
    data = [word.lower() for word in tokens if word.lower() not in stopwords.words('english')
            and word not in string.punctuation and word[0] not in string.punctuation and word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(word) for word in data]
    return data

train_data = preprocess(train_data)
val_data = preprocess(val_data)

# Unigram
class UnigramLanguageModel:
    def __init__(self, data, unknown_words_method='k', k=0):
        self.data = data
        self.words_count = 0
        self.unknown_words_method = unknown_words_method
        self.k = k
        self.counts = defaultdict(int)
        self.probs = defaultdict(float)
        self.word_set = set()
        self.val_data = None
        self.val_counts = defaultdict(int)
        self.unknown_word_count = 0

    def calculate_wordcounts(self, dataset_type='train'):
        """
    Count word occurrences in the specified dataset.

    Parameters:
        - dataset_type (str): Either 'train' for the training dataset or 'val' for the validation dataset.

    This method counts the occurrences of words in the given dataset and updates the internal counts and word set.
    If 'val' is specified, it also handles unknown words by marking them as '<UNK>'.
    """
        if dataset_type == 'train':
            for word in self.data:
                self.words_count += 1
                self.counts[word] += 1
        elif dataset_type == 'val':
            for word in self.val_data:
                if word not in self.word_set:
                    self.unknown_word_count += 1
                    word = '<UNK>'
                self.val_counts[word] += 1

    def calculate_wordprobabilities(self):
        """
    Calculate word probabilities based on word counts.

    This method computes the probabilities of each word in the vocabulary based on their counts in the training data.
    The probabilities are stored in the 'probs' dictionary.
    """
        total_word_count = sum(self.counts.values())
        for word, count in self.counts.items():
            self.probs[word] = count / total_word_count


    def update_wordprobabilities_with_val(self):
        """
    Update word probabilities with validation set counts.

    This method updates the word probabilities based on counts from the validation set.
    The updated probabilities are stored in the 'probs' dictionary.
    """
        if self.unknown_words_method == 'unk':
            self.probs['<UNK>'] = (self.unknown_word_count) / (self.words_count)
        elif self.unknown_words_method == 'k':
            for word, val_count in self.val_counts.items():
                if word not in self.word_set:
                    self.probs[word] = (val_count + self.k) / (self.words_count + self.k * len(self.counts))

    def perplexity_Unigram(self):
        """
        Calculate perplexity for a unigram language model on a test corpus.

        The perplexity is computed using the following formula:
        PP = exp((1/N) * Σ(-log P(wi))) where N is the total number of tokens
        in the test corpus and P(wi) is the unigram probability of the word wi
        based on the trained unigram model.

        Returns:
            float: The perplexity score for the unigram model on the test corpus.
        """
        def entropy(dictvalues):
            totalentropy = 0
            for key, values in dictvalues.items():
                totalentropy += (-1) * math.log(values, 2)
            totalentropy /= len(dictvalues)
            return totalentropy

        return math.pow(2, entropy(self.probs))

    def train(self):
        self.calculate_wordcounts()
        self.word_set = set(self.counts.keys())
        self.calculate_wordprobabilities()

    def validate(self, val_data):
        self.val_data = val_data
        self.calculate_wordcounts('val')
        self.update_wordprobabilities_with_val()



# Unigram without smoothing
unigram_train_no_smoothing = UnigramLanguageModel(train_data)
unigram_train_no_smoothing.train()

unigram_val_no_smoothing = UnigramLanguageModel(val_data)
unigram_val_no_smoothing.train()

print('Train data: Perplexity of Unigram (No Smoothing):', unigram_train_no_smoothing.perplexity_Unigram())
print('validation data: Perplexity of Unigram (No Smoothing):', unigram_val_no_smoothing.perplexity_Unigram())

# Smoothed - Laplace smoothing
unigram_train_laplace = UnigramLanguageModel(train_data, 'k', 1)
unigram_train_laplace.train()

unigram_val_laplace = UnigramLanguageModel(val_data, 'k', 1)
unigram_val_laplace.train()
unigram_val_laplace.validate(train_data)  # Validate using the training data

# print('Perplexity of Unigram using Laplace smoothing on train data:', unigram_train_laplace.perplexity_Unigram())
print('validation data: Perplexity of Unigram using Laplace smoothing:', unigram_val_laplace.perplexity_Unigram())

# Smoothed - Add-k smoothing with k=0.5
unigram_train_addk = UnigramLanguageModel(train_data, 'k', 0.5)
unigram_train_addk.train()

unigram_val_addk = UnigramLanguageModel(val_data, 'k', 0.5)
unigram_val_addk.train()
unigram_val_addk.validate(train_data)  # Validate using the training data

# print('Perplexity of Unigram using Add-k smoothing (k=0.5) on train data:', unigram_train_addk.perplexity_Unigram())
print('validation data: Perplexity of Unigram using Add-k smoothing (k=0.5)', unigram_val_addk.perplexity_Unigram())


unigram_val_unknown = UnigramLanguageModel(val_data, 'unk', 1)
unigram_val_unknown.train()
print('validation data: Perplexity of Unigram after unknown word handling',  unigram_val_unknown.perplexity_Unigram())

# # Initialize unigram models with different smoothing methods for training
# unigram_train_no_smoothing = UnigramLanguageModel(train_data)
# unigram_train_laplace = UnigramLanguageModel(train_data, 'k', 1)
# unigram_train_addk = UnigramLanguageModel(train_data, 'k', 0.5)
# unigram_train_unk = UnigramLanguageModel(train_data, 'unk', 0)

# # Train unigram models
# unigram_train_no_smoothing.train()
# unigram_train_laplace.train()
# unigram_train_addk.train()
# unigram_train_unk.train()

# # Initialize unigram models with different smoothing methods for validation
# unigram_val_no_smoothing = UnigramLanguageModel(val_data)
# unigram_val_laplace = UnigramLanguageModel(val_data, 'k', 1)
# unigram_val_addk = UnigramLanguageModel(val_data, 'k', 0.5)
# unigram_val_unk = UnigramLanguageModel(val_data, 'unk', 0)

# # Validate unigram model
# # unigram_val_no_smoothing.validate(train_data)
# # unigram_val_laplace.validate(train_data)
# # unigram_val_addk.validate(train_data)
# # unigram_val_unk.validate(train_data)

# # unigram_val_no_smoothing.validate(val_data)
# # unigram_val_laplace.validate(val_data)
# # unigram_val_addk.validate(val_data)
# # unigram_val_unk.validate(val_data)

# unigram_val_no_smoothing.train()
# unigram_val_laplace.train()
# unigram_val_addk.train()
# unigram_val_unk.train()

# # Calculate perplexity for training and validation sets
# perplexity_train_no_smoothing = unigram_train_no_smoothing.perplexity_Unigram()
# perplexity_val_no_smoothing = unigram_val_no_smoothing.perplexity_Unigram()

# perplexity_train_laplace = unigram_train_laplace.perplexity_Unigram()
# perplexity_val_laplace = unigram_val_laplace.perplexity_Unigram()

# perplexity_train_addk = unigram_train_addk.perplexity_Unigram()
# perplexity_val_addk = unigram_val_addk.perplexity_unigram()

# perplexity_train_unk = unigram_train_unk.perplexity_unigram()
# perplexity_val_unk = unigram_val_unk.perplexity_unigram()

# # Print perplexity values
# print("Perplexity of Unigram on training data (No Smoothing):", perplexity_train_no_smoothing)
# print("Perplexity of Unigram on validation data (No Smoothing):", perplexity_val_no_smoothing)
# print("Perplexity of Unigram using Laplace smoothing on training data:", perplexity_train_laplace)
# print("Perplexity of Unigram using Laplace smoothing on validation data:", perplexity_val_laplace)
# print("Perplexity of Unigram using Add-k smoothing (k=0.5) on training data:", perplexity_train_addk)
# print("Perplexity of Unigram using Add-k smoothing (k=0.5) on validation data:", perplexity_val_addk)
# print("Perplexity of Unigram after unknown word handling on training data:", perplexity_train_unk)
# print("Perplexity of Unigram after unknown word handling on validation data:", perplexity_val_unk)


# Bigram
from collections import Counter

class BigramLanguageModel:
    def __init__(self, data, unknown_words_method='k', k=0):
        self.data = data
        self.words_count = 0
        self.unknown_words_method = unknown_words_method
        self.k = k
        self.counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.bigram_probs = defaultdict(float)
        self.bigrams = []
        self.unknown_words_count = 0
        self.val_data = None
        self.val_bigrams = []
        self.val_counts = defaultdict(int)
        self.val_bigram_counts = defaultdict(int)

    def calculate_word_counts_bigram(self, dataset_type='train'):
        """
    Calculate word and bigram counts for either training or validation data.

    This function counts individual words and bigrams in the given data based on
    the specified 'train_or_val' mode. It updates word counts, bigram counts, and
    handles unknown words if applicable.

    Args:
        train_or_val (str, optional): The mode to determine whether to count
            words and bigrams for training or validation data ('train' or 'val').
            Defaults to 'train'.

    """
        if dataset_type == 'train':
            self.bigrams = list(zip(self.data, self.data[1:]))
            for word in self.data:
                self.words_count += 1
                self.counts[word] += 1
            for bigram in self.bigrams:
                self.bigram_counts[bigram] += 1
        elif dataset_type == 'val':
            self.val_bigrams = list(zip(self.val_data, self.val_data[1:]))
            for word in self.val_data:
                if word not in self.word_set:
                    self.unknown_words_count += 1
                    word = '<UNK>'
                self.counts[word] += 1
            for bigram in self.val_bigrams:
                w1, w2 = bigram
                if w1 not in self.word_set:
                    w1 = '<UNK>'
                if w2 not in self.word_set:
                    w2 = '<UNK>'
                self.val_bigram_counts[(w1, w2)] += 1

    def calculate_word_probabilities_bigram(self):
        """
    Calculate bigram probabilities based on word counts.

    This function calculates bigram probabilities using the counts of bigrams
    and the counts of their first words. It updates the bigram probabilities
    dictionary.
    """
        for word, count in self.bigram_counts.items():
            self.bigram_probs[word] = (count) / (self.counts[word[0]])

    def update_word_probabilities_with_val_bigram(self):
        """
    Update bigram probabilities with validation data.

    This function updates bigram probabilities based on the counts of bigrams
    from the validation data and the counts of their first words in the training
    data. It applies the specified unknown word handling method (UNK or Add-k
    smoothing) to account for unseen bigrams during validation.
    """
        if self.unknown_words_method == 'unk':
            self.bigram_probs[('<UNK>', '<UNK>')] = self.bigram_counts[('<UNK>', '<UNK>')] / self.unknown_words_count
        elif self.unknown_words_method == 'k':
            for word, val_count in self.val_bigram_counts.items():
                if word not in self.bigram_counts:
                    self.bigram_probs[word] = (val_count + self.k) / (
                            self.counts[word[0]] + self.k * len(self.counts))

    def perplexity_bigram(self):
        """
        Calculate perplexity for a bigram language model on a test corpus.

        The perplexity is computed using the following formula:
        PP = exp((1/N) * Σ(-log P(wi|wi-1))) where N is the total number of tokens
        in the test corpus and P(wi|wi-1) is the bigram probability of the word wi
        given the previous word wi-1, based on the trained bigram model.

        Returns:
            float: The perplexity score for the bigram model on the test corpus.
        """
        def entropy(dictvalues):
            totalentropy = 0
            for key, values in dictvalues.items():
                if values == 0:
                    continue
                totalentropy += (-1) * math.log(values, 2)
            totalentropy /= len(dictvalues)
            return totalentropy

        return math.pow(2, entropy(self.bigram_probs))

    def train(self):
        self.calculate_word_counts_bigram()
        self.word_set = set(self.counts.keys())
        self.calculate_word_probabilities_bigram()

    def validate(self, val_data):
        self.val_data = val_data
        self.calculate_word_counts_bigram('val')
        self.calculate_word_probabilities_bigram()

# Bigram without smoothing
bigram_train_no_smoothing = BigramLanguageModel(train_data)
bigram_train_no_smoothing.train()

# Bigram without smoothing
bigram_val_no_smoothing = BigramLanguageModel(val_data)
bigram_val_no_smoothing.train()

print('Train data : Perplexity of Bigram (No Smoothing):', bigram_train_no_smoothing.perplexity_bigram())
print('validation data : Perplexity of Bigram (No Smoothing):', bigram_val_no_smoothing.perplexity_bigram())
# Smoothed - Laplace smoothing for Bigram
bigram_train_laplase = BigramLanguageModel(train_data)
bigram_train_laplase.train()

bigram_val_laplase = BigramLanguageModel(val_data, 'k', 1)
bigram_val_laplase.train()
bigram_val_laplase.validate(train_data)  # Validate using the training data

print('validation data: Perplexity of Bigram using Laplace smoothing:', bigram_val_laplase.perplexity_bigram())

# Smoothed - Add-k smoothing with k=0.5 for Bigram
bigram_val_addk = BigramLanguageModel(val_data, 'k', 0.5)
bigram_val_addk.train()
bigram_val_addk.validate(train_data)  # Validate using the training data

print('Perplexity of Bigram using Add-k smoothing (k=0.5) on validation data:', bigram_val_addk.perplexity_bigram())

bigram_val_unknown = BigramLanguageModel(val_data, 'unk', 1)
bigram_val_unknown.train()
print('Perplexity of Bigram after unknown word handling on validation data:',  bigram_val_unknown.perplexity_bigram())
