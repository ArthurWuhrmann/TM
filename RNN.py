import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
from torch.utils import data

class Functs:

    #Renvoie  f(x) où f est la fonction sigmoïde
    def sig(x):
        return 1/(1+np.exp(-x))

    #Renvoie f(x) où f est la dérivée de la fonction sigmoïde
    def d_sig(x):
        x_ = sig(x)
        return x_ * ( 1 - x_)

    #Renvoie f(x) où f est la fonction tangeante hyperbolique
    def tanh(x):
        return (1 - np.exp(-2*x))/(1 + np.exp(-2*x))

    #Renvoie f(x) où f est la dérivée de la fonction tangeante hyperbolique
    def d_tanh(x):
        return 1 - Functs.tanh(x)**2

    #Renvoie le softmax d'un vecteur v
    def softmax(v):
        return np.exp(v) / sum(np.exp(v))
        



class Dataset(data.Dataset): # ------ Pas sûr de garder, pour le moment histoire de voir son utilité

    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        X = self.inputs[index]
        y = self.targets[index]

        return X, y

#Génère n 'phrases' générées selon le modèle "Début" + k * "x" + k * "y" + "Fin"
#pour k aléatoire ensemble de [0; 10]
def gen_data(n=100):

    datas = []

    for i in range(n):
        random = np.random.randint(1, 10)

        data = ['BOS'] + ['x']*random + ['y']*random + ['EOS']

        datas.append(data)

    return datas


def seq_to_dicts(seq):

    all_ = []

    for sent in seq:
        for word in sent:
            all_.append(word) #On parcourt chaque mot de chaque phrase et on l'ajoute a 'all_'
    
    word_count = defaultdict(int) #Initialisation d'un dictionnaire vide, qui contiendra les fréquences

    for word in all_:
        word_count[word] += 1 #La boucle parcourt les mots et pour chaque itération elle incrémente la fréquence du mot


    unique_words = list(word_count.keys())
    #On fait une liste avec chaque mot du dictionnaire

    unique_words.append("UNKNOWN")
    #C'est utile si on décide de faire apprendre que les mots les plus fréquents. Comme cela,
    #les mots moins fréquents seront remplacés par 'UNKNOWN'. ça permet d'optimiser le temps d'apprentissage
    #pour les textes complexes.

    n_sentences, vocab_size = len(seq), len(unique_words)
    #La première valeur contient le nombre de phrases qu'on a donné à la fonction, et la deuxième la
    #taille du vocabulaire (le nombre de mots différents + 1 pour 'UNKNOWN' ici)
    
    word_to_index = defaultdict(lambda: vocab_size-1) #Renvoie par défaut l'index de 'UNKNOWN'
    index_to_word = defaultdict(lambda: 'UNKNOWN') #Renvoie par défaut le mot 'UNKNOWN'

    for index, word in enumerate(unique_words):

        word_to_index[word] = index
        index_to_word[index] = word

        #On remplit les deux dictionnaires

    return n_sentences, vocab_size, word_to_index, index_to_word
    

class RNN:

    def __init__(self, hidden_size, sequences, sequences_params):

        #On initialise tous les vecteurs de neurones (entrée, cachés, sortie)

        self.sequences = sequences

        self.vocab_size = vocab_size

        self.hidden_size = hidden_size

        self.n_sentences, self.vocab_size, self.word_to_index, self.index_to_word = sequences_params

        self.training_set, self.validation_set, self.test_set = self.create_datasets(Dataset)

    #Génère n 'phrases' générées selon le modèle "Début" + k * "x" + k * "y" + "Fin"
    #pour tout k ensemble de [0; 10]
    def gen_data(n=100):

        datas = []

        for i in range(n):
            random = np.random.randint(1, 10)

            data = ['x']*random + ['y']*random + ['EOS']

            datas.append(data)

        return datas

    #Répartit les valeurs d'une manière particulière, cf. https://arxiv.org/abs/1312.6120
    def init_orthogonal(param):

        rows, cols = param.shape

        param_ = np.random.randn(rows, cols)

        if rows < cols:
            param_ = param_.T

        q, r = np.linalg.qr(param_)

        ph = np.sign(np.diag(r))

        q *= ph

        if rows < cols:

            q = q.T

        return q


    #Cette fonction permet d'éviter que les gradients ne deviennent trop grand
    #(c.f.: exploding gradients problems). On va prendre la norme de tout les gradients
    #puis calculer le coefficient avec le paramètre max_norm. Si il est inférieur
    #à 1 on ne fait rien mais sinon on réduit les gradients
    def clip_grads(grads, max_norm=.25):

        total_norm = 0

        for grad in grads:
            total_norm += np.sum(np.power(grad, 2))

        total_norm = np.sqrt(total_norm)

        coeff = max_norm/total_norm

        if coeff < 1:
            for grad in grads:

                grad*=coeff

        return grads

    #Initalises les paramètres du réseau
    def init_rnn(self):

        U = np.random.randn(self.hidden_size, self.vocab_size)
        U = RNN.init_orthogonal(U)

        V = np.random.randn(self.hidden_size, self.hidden_size)
        V = RNN.init_orthogonal(V)

        W = np.random.randn(self.vocab_size, self.hidden_size)
        W = RNN.init_orthogonal(W)

        bias_hidden = np.zeros((self.hidden_size, 1))

        bias_out = np.zeros((self.vocab_size, 1))

        self.params = U, V, W, bias_hidden, bias_out

        return self.params

    #Effectue un passage feed forward
    def forward(self, inputs, hidden_t):

        U, V, W, bias_hidden, bias_out = self.params
        outputs, hidden_states = [], []

        for t in range(len(inputs)): #Boucle pour chaque lettre...

            hidden_t = Functs.tanh(np.dot(U, inputs[t])
                                   + np.dot(V, hidden_t)
                                   + bias_hidden)

            out = Functs.softmax(np.dot(W, hidden_t) + bias_out)
            
            #out est un vecteur de même taille que le vocabulaire, qui contient la
            #probabilité de chaque lettre d'apparaître (ici la lettre après input[t]

            outputs.append(out)

            hidden_states.append(hidden_t)

        return outputs, hidden_states


    #Effectue une rétro-propagation
    def backward(self, inputs, outputs, hidden_states, targets):

        #Initialisation des valeurs dont on aura besoin

        U, V, W, bias_hidden, bias_out = self.params

        d_U, d_V, d_W = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W)

        d_bias_hidden, d_bias_out = np.zeros_like(bias_hidden), np.zeros_like(bias_out)

        d_h_next = np.zeros_like(hidden_states[0])

        loss = 0 #On définit le coût égal à 0 à l'origine

        for t in reversed(range(len(outputs))): #On parcourt le réseau à l'envers

            loss += -np.mean(np.log(outputs[t]) * targets[t]) #On calcule le coût avec une entropie croisée (plus pratique que la somme des écarts à la moyenne)

            d_o = outputs[t].copy()
            d_o[np.argmax(targets[t])] -= 1

            d_W += np.dot(d_o, hidden_states[t].T)
            d_bias_out += d_o

            d_h = np.dot(W.T, d_o) + d_h_next

            d_f = Functs.d_tanh(hidden_states[t]) * d_h

            d_bias_hidden += d_f

            d_U += np.dot(d_f, inputs[t].T)

            d_V += np.dot(d_f, hidden_states[t-1].T)

            d_h_next = np.dot(V.T, d_f)

        grads = d_U, d_V, d_W, d_bias_hidden, d_bias_out

        clipped_grads = RNN.clip_grads(grads) #On clippe les gradients pour pas qu'il soient trop grands

        return loss, grads

    #Modifie les paramètres en fonction des gradients passés dans l'appel de la fonciton

    def update_parameters(self, grads, lr=1e-3):
        for param, grad in zip(self.params, grads):
            param -= lr * grad
    
        return params

    #Répartit l'intégralité des données dans les ensemble d'entraînement, de validation et de test 

    def create_datasets(self, dataset_class, p_train=0.8, p_validation =.1, p_test=.1):

        n_train = int(len(self.sequences)*p_train)
        n_validation = int(len(self.sequences)*p_validation)
        n_test = int(len(self.sequences)*p_test)

        sequences_train = self.sequences[:n_train]
        sequences_validation = self.sequences[n_train:n_train+n_validation]
        sequences_test = self.sequences[-n_test:]

        def seq_to_in_and_targets(seqs):

            inputs, targets = [], []

            for seq in seqs:

                inputs.append(seq[:-1])
                targets.append(seq[1:])

            return inputs, targets

        training = seq_to_in_and_targets(sequences_train)
        validation = seq_to_in_and_targets(sequences_validation)
        test = seq_to_in_and_targets(sequences_test)

        training_set = dataset_class(training[0], training[1])
        validation_set = dataset_class(validation[0], validation[1])
        test_set = dataset_class(test[0], test[1])

        return training_set, validation_set, test_set

def one_hot_encode(index, vocab_size):

    #Cette fonction permet d'encoder un caractère
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec


def one_hot_encode_seq(sequence, vocab_size, word_to_index):

    #Cette fonction permet d'encoder une phrase
    
    vec = np.array([one_hot_encode(word_to_index[word], vocab_size) for word in sequence])

    vec = vec.reshape(vec.shape[0], vec.shape[1], 1)

    return vec

sequences = gen_data(50) #Cette variable sert juste à obtenir les vocs et la taille... les données
#ne sont pas directement utilisés

hidden_size = 100 #Taille de la couche cachée du réseau

sequences_params = seq_to_dicts(sequences) #On récupère toutes les spécificités des séquences (vocabulaire, taille, etc.)

n_sentences, vocab_size, word_to_index, index_to_word = sequences_params #On unpack ces paramètres

rnn = RNN(hidden_size, sequences, sequences_params) #Création d'un objet de la classe RNN, un réseau de neurones récurrents

training_set, validation_set, test_set = rnn.create_datasets(Dataset)

num_epochs = 1000 #Nombres de fois où on va entrainer puis valider tout le réseau

tstart = time.time() #On lance un timer pour savoir combien de temps le réseau met à faire ses calculs

params = rnn.init_rnn()

name = "rnn_tm"

hidden_state = np.zeros((hidden_size, 1))


training_loss, validation_loss = [], []

lr = 1e-2

for i in range(num_epochs+1):
    
    epoch_training_loss = 0
    epoch_validation_loss = 0

    for inputs, targets in validation_set:

        #On encode les phrases en en vecteurs "one-hot" (donc avec un 1 et pleins de 0)
        inputs_one_hot = one_hot_encode_seq(inputs, vocab_size, word_to_index)
        targets_one_hot = one_hot_encode_seq(targets, vocab_size, word_to_index)
        
        hidden_state = np.zeros_like(hidden_state)

        
        outputs, hidden_states = rnn.forward(inputs_one_hot, hidden_state)


        loss, _ = rnn.backward(inputs_one_hot, outputs, hidden_states, targets_one_hot)

        epoch_validation_loss += loss

    #On parcout l'ensemble d'entraînement
    for inputs, targets in training_set:
        
        inputs_one_hot = one_hot_encode_seq(inputs, vocab_size, word_to_index)
        targets_one_hot = one_hot_encode_seq(targets, vocab_size, word_to_index)
        
        hidden_state = np.zeros_like(hidden_state)

        outputs, hidden_states = rnn.forward(inputs_one_hot, hidden_state)

        loss, grads = rnn.backward(inputs_one_hot, outputs, hidden_states, targets_one_hot)
        
        if np.isnan(loss):
            raise ValueError('Gradients have vanished!')
        
        params = rnn.update_parameters(grads, lr=1e-3)
        
        epoch_training_loss += loss
        
    training_loss.append(epoch_training_loss/len(training_set)) #On garde les coûts pour les afficher après avec matplotlib
    validation_loss.append(epoch_validation_loss/len(validation_set))

    if i % 100 == 0:
        i = max(1, i) #Pour commer à l'époque 1
        #lr = lrbase * 100/i
        print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

        #Affichage matplotlib
        
        epoch_fig = np.arange(len(training_loss))
        plt.figure()
        plt.plot(epoch_fig, training_loss, 'r', label="Erreur (phase d'entraînement)",)
        plt.plot(epoch_fig, validation_loss, 'b', label="Erreur (phase de validation)")
        plt.legend()
        plt.xlabel('Époque'), plt.ylabel('Erreur')
        path = name + "/" + str(epoch_fig[-1:][0])
        if not os.path.isdir(path):
            os.makedirs(path)
        delay = time.time()-tstart
        delay = float(int(delay*1e3))/1e3
        delay = str(delay)[:str(delay).find(".")+2].replace(".", "_")
        if len(delay) < 3:
            delay = "first"
        print(delay.replace("_", "."))

        plt.ylim(0, 5)

        #On enregistre une photo du graphe
        
        plt.savefig(name + "/" + str(epoch_fig[-1:][0]) + "/t" + delay)
        plt.close()
