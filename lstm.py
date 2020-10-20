import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import glob
import mido
import time
from figure_threads import ShowFigure


#Cette calsse s'occupe de gérer ce qui est en lien avec la musique mais qui peut être fait en dehors de la classe LSTM
class Music:

    #Génère des données à étudier. Ici, la méthode va chercher les n premiers fichiers .mid dans le dossier /midis/ par rapport au fichier, pour n passé en paramètre.
    def gen_datas(n):

        dic = {}

        t1 = time.time()
        
        datas = []
        text_datas = []

        for index, midfile in enumerate(os.listdir('midis')):
            if index < n:
                text_datas += [Music.midi_to_text('midis/'+midfile)]
            else:
                break

        unique_chars = set()
        for index, text_data in enumerate(text_datas):
            for char in text_data:
                
                unique_chars.add(char)

        for index, char in enumerate(unique_chars):
            dic[char] = index
            dic[index] = char

        encoded_datas = []
        for index, text_data in enumerate(text_datas):
            encoded_datas += Music.encode_text(text_data, dic)

        np.random.shuffle(encoded_datas)
        t2 = time.time()
        
        print('Traitement des données terminé.', t2-t1,'sec.')

        return encoded_datas, dic


    #Prend un .mid en entrée et renvoie un chaîne de caractère. Ce code peut être changé si l'on veut par exemple transcrire la durée des notes différemment

    def midi_to_text(midifile):
        mid = mido.MidiFile(midifile)
        
        tempo = 493827
        
        tpb = mid.ticks_per_beat
        
        text = ""
        max_ = 500
        index = 0
        for msg in mid:
            if msg.is_meta and msg.type =='set_tempo':
                tempo = msg.tempo
            if msg.type == 'note_on':
                t = int(mido.second2tick(msg.time, tpb, tempo))
                if t != 0:
                    t = chr(200)
                else:
                    t = ''
                text += chr(33 + msg.note)
            index += 1
            if index == max_:
                pass
                #break

        return text

    #Permet de séparer un morceau en plusieurs séquence, que l'on passera au réseau.
    def encode_text(text, dic):

        vocab_size = len(dic)//2
        seq = []
        seqs = []
        for char in text:
            seq.append(Utils.one_hot_encode(vocab_size, dic[char[0]]))

        for i in range(0, len(seq), 10):
            seqs.append(seq[i:i+20])    
        
        return [seq] 

#Finie
class Utils:


    #Génère n données comme suivant schéma "Début" + k* "X" + k*"Y" + "Fin" pour n passé en paramètre et k aléatoire entre 0 et 10
    def gen_datas(n):

        datas = []
        dic = {}
        unique_words = set()

        for i in range(n):
            xy = np.random.randint(3, 10)
            datas += [xy * ['x'] + xy * ['y'] + ['EOS']]


        for sentence in datas:
            for word in sentence:
                unique_words.add(word)
        size = len(unique_words)


        for unique_word, index in zip(unique_words, range(size)):
            vec = Utils.one_hot_encode(size, index)
            dic[unique_word] = index
            dic[index] = unique_word

        return datas, dic

    #Permet de créer un vecteur nul sauf à l'index passé en paramètre, où il vaut 1.

    def one_hot_encode(size, index):

        # ! Attention, index à partir de 0 !
        
        vec = np.zeros((size, 1))
        vec[index] = 1

        return vec

    #Permet d'encoder une suite de caractère selon la méthode one_hot_encode ou chaque caractère différent à un index respectif.

    def sent_hot_encode(sentence, dic):

        size = len(dic)//2 #Il faut diviser par deux car le dictionnaire
        #contient les "traductions" dans les deux sens.

        encode = []
        for letter in sentence:
            assert letter in dic #A enlever à la fin
            encode.append(Utils.one_hot_encode(size, dic[letter]))

        arr = np.array(encode)
        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)

        return arr

    #Renvoie l'index de la valeur la plus élevée du vecteur passé en paramètre. Permet d'obtenir la valeur avec le plus de probabilités d'apparaître au procain temps selon le réseau.

    def vec_to_index(vec):
        return np.argmax(vec)

    #Partitionne les données

    def part(datas, parts):
        parts_sum = np.sum(parts)
        p_training, p_validation, p_testing = parts
        p_training /= parts_sum
        p_validation /= parts_sum
        p_testing /= parts_sum

        def compute(datas, part, index):

            n = int(len(datas)*part)
            print(n)

            seqs = datas[index:index+n]

            inputs, targets = [], []

            for seq in seqs:
                inputs.append(seq[:-1])
                targets.append(seq[1:])

            return ((inputs, targets), n)

        training_set, index = compute(datas, p_training, 0)
        validation_set, index = compute(datas, p_validation, index)
        testing_set, _ = compute(datas, p_testing, index)

        return training_set, validation_set, testing_set


    #Fonction sigmoïde
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    #Dérivée de la sigmoïde
    def d_sigmoid(x):
        y = Utils.sigmoid(x)
        return y * (1-y)

    #Tangeante hyperbolique
    def tanh(x):
        
        return np.tanh(x)

    #Dérivée de la tangeante hyperbolique
    def d_tanh(x):
        return 1-Utils.tanh(x)**2

    #Fonction Softmax, permet de répartir les valeurs d'un vecteur de manière à ce que la somme de ces dernières soit comprise entre 0 et 1.
    def softmax(v):
        return np.exp(v) / sum(np.exp(v))

    # c.f https://arxiv.org/abs/1312.6120, permet d'optimiser le réseau.
    def init_orthogonal(shape):

        rows, cols = shape
        param = np.random.randn(rows, cols)
        transform = rows < cols
        
        if transform:
            param = param.T

        q, r = np.linalg.qr(param)
        ph = np.sign(np.diag(r))
        q *= ph

        if transform:
            return q.T
        
        return q

    #Fonction qui calcule le coût du réseau (ici entropie croisée)
    def loss(outputs, targets):
        return -np.mean(np.log(outputs) * targets)

    #Fonction Dérivée de ce coût
    def d_loss(outputs, targets):
        
        return outputs - targets

    #Fonction qui renvoie le taux d'apprentissage pour une époque donnée, permet de faire varier ce dernier. On peut ainsi le faire réduire au fur et à mesure que le réseau progresse.
    def learning_rate(epoch_index):

        return 1e-02


class LSTM:

    #Fini (?)
    def __init__(self, hidden_size, name="lstm"):

        self.hidden_size = hidden_size
        self.name = name

    #Fini
    def init_params(self, lr=0.01):

        weight_forget = Utils.init_orthogonal((hidden_size, self.z_size)) #(10, 20)
        weight_input =  Utils.init_orthogonal((hidden_size, self.z_size)) #(10, 20)
        weight_g = Utils.init_orthogonal((hidden_size, self.z_size)) #(10, 20)
        weight_out = Utils.init_orthogonal((hidden_size, self.z_size)) #(10, 20)
        weight_v = Utils.init_orthogonal((self.vocab_size, hidden_size)) #(10, 10)

        bias_forget = np.zeros((hidden_size, 1))
        bias_input = np.zeros((hidden_size, 1))
        bias_g = np.zeros((hidden_size, 1))
        bias_out = np.zeros((hidden_size, 1))
        bias_v = np.zeros((self.vocab_size, 1))

        weights = weight_forget, weight_input, weight_g, weight_out, weight_v
        biases = bias_forget, bias_input, bias_g, bias_out, bias_v
        self.params = weights + biases

        self.learning_rate = lr

    #Affiche les prédictions du réseau ainsi que les prédiction "justes" en fonction de l'entrée donnée
    def print_prediction(self, input_, target, dic):

        forward_outputs = self.forward(input_)

        print("Phrase d'entrée : ")
        print([dic[np.argmax(letter)] for letter in input_])

        print("Phrase cible : ")
        print([dic[np.argmax(letter)] for letter in target])

        print("Phrase calculée : ")
        
        print([dic[np.argmax(output)] for output in forward_outputs[-1:][0]])

    #Sauvegarde dans un fichier les paramètres actuels du réseau, qui peuvent être chargés grace à load_network
    def save_params(self, index_epoch):

        #Paramètres à sauvegarder :
        #self.params (les biais, les poids, etc.)
        #les tailles du réseau (hidden_size, vocab_size)

        path = self.name + "/" + str(index_epoch)

        if not os.path.isdir(path):
            os.makedirs(path)

        data = {}

        data['fields'] = []
        data['fields'] = {
            'name':self.name,
            'hidden_size':self.hidden_size,
            'vocab_size':self.vocab_size
            }

        with open(path + "/network_params.json", "w") as file:
            json.dump(data, file)

        if not os.path.isdir(path + "/params"):
            os.makedirs(path + "/params")

        for i in range(len(self.params)):
            np.savetxt((path + "/params/param_" + str(i) + ".txt"), self.params[i])

    #Charge un réseau a partir de paramètres stockés dans un fichier grâce à NumPy    
    def load_network(name, epoch):

        hidden_size, vocab_size = 0, 0

        path = name + "/" + str(epoch)

        params = ()
        
        with open(path + "/network_params.json") as json_file:

            data = json.load(json_file)
            hidden_size = data['fields']['hidden_size']
            vocab_size = data['fields']['vocab_size']
            

            for path in glob.glob(path + "/params/*.txt"):

                params += (np.loadtxt(path),)

        network = LSTM((vocab_size, hidden_size), name)
        network.params = params
        return network

    #Permet d'éviter que les gradients ne soient trop grands ou trop petits
    def clip_gradients(self, max_norm=.25):

        total_norm = 0
        for grad in self.grads:
            total_norm += np.sum(np.power(grad, 2))

        total_norm = np.sqrt(total_norm)
        coeff = max_norm/total_norm

        if coeff > 1:
            for grad in self.grads:
                grad *= coeff

    #Modifie les paramètres du réseau selon les gradients donnés en paramètres de la méthode
    def update_parameters(self, learning_rate=.01):

        for param, grad in zip(self.params, self.grads):
            param -= learning_rate * grad
        

    #Effectue un passage feedforward
    def forward(self, inputs, hidden_previous=np.zeros((0,0)), cell_previous=np.zeros((0,0))):

        if hidden_previous.shape == (0,0): #Ces lignes permettent de ne pas avoir
            hidden_previous = np.zeros((self.hidden_size, 1)) #besoin de passer en
        if cell_previous.shape == (0,0): # paramètres les couches cachées et les
            cell_previous = np.zeros((self.hidden_size, 1)) #cellules. Petite optimisation.

        input_gates, forget_gates, out_gates, candidate_gates = [], [], [], []

        z_values = []

        weight_forget, weight_input, weight_g, weight_out, weight_v = self.params[:5]
        bias_forget, bias_input, bias_g, bias_out, bias_v = self.params[-5:]

        hiddens = [hidden_previous]
        cells = [cell_previous]

        output_values, outputs = [], []

        for i in range(len(inputs)):

            x = inputs[i]

            z_t = np.row_stack((hidden_previous, x)) #Concaténation du hidden_t précédant
            #et de l'entrée à t.
            z_values.append(z_t)

            input_gate = Utils.sigmoid(np.dot(weight_input, z_t) + bias_input) #Calcul de la porte d'entrée
            input_gates.append(input_gate)

            forget_gate = Utils.sigmoid(np.dot(weight_forget, z_t)) #Calcul de la porte d'oubli
            forget_gates.append(forget_gate)

            candidate_gate = Utils.tanh(np.dot(weight_g, z_t) + bias_g) #Calcul de la porte "candidate"
            candidate_gates.append(candidate_gate)

            cell_previous = forget_gate * cell_previous + input_gate * candidate_gate #Calcul de la cellule
            #de "mémoire" à l'instant t
            cells.append(cell_previous)

            out_gate = Utils.sigmoid(np.dot(weight_out, z_t)) #Calcul de la porte de sortie
            out_gates.append(out_gate)

            hidden_previous = Utils.tanh(cell_previous) * out_gate #Calcul de la couche cachée au temps t
            hiddens.append(hidden_previous)

            output_value = np.dot(weight_v, hidden_previous) + bias_v #Calcul la valeur de sortie
            output_values.append(output_value)

            output = Utils.softmax(output_value) #Renvoie la probabilité d'apparition de chaque élément du vocabulaire
            outputs.append(output)
        
        return hiddens, cells, input_gates, forget_gates, out_gates, candidate_gates, z_values, output_values, outputs

    #La fonction backward renvoie également le loss, mais dans la partie "validation", nous n'avons
    #besoin de calculer les gradients, nous utilisons la partie de validation uniquement pour calculer le loss.
    #Créer une autre fonction le calculant permet d'optimiser légèrement le code.
    def compute_loss(packed_params, targets):
        
        loss = 0
        z_values, output_values, outputs = packed_params[-3:]

        for t in reversed(range(len(outputs))): #Pas besoin de parcourir dans le sens inverse,
            #nous n'avons besoin que de l'output et du target à chaque temps t, que nous possédons déjà.

            loss += Utils.loss(outputs[t], targets[t])
            
        return loss

    #Effectue une rétro-propagation sur les paramètres du réseau afin de calculer les gradients
    def backward(self, packed_params, targets):

        weight_forget, weight_input, weight_g, weight_out, weight_v = self.params[:5]
        bias_forget, bias_input, bias_g, bias_out, bias_v = self.params[-5:]

        hiddens, cells, input_gates, forget_gates, out_gates, candidate_gates = packed_params[:6]
        z_values, output_values, outputs = packed_params[-3:]

        dweight_forget = np.zeros_like(weight_forget)
        dweight_input = np.zeros_like(weight_input)
        dweight_out = np.zeros_like(weight_out)
        dweight_v = np.zeros_like(weight_v)
        dweight_g = np.zeros_like(weight_g)

        dbias_forget = np.zeros_like(bias_forget)
        dbias_input = np.zeros_like(bias_input)
        dbias_out = np.zeros_like(bias_out)
        dbias_v = np.zeros_like(bias_v)
        dbias_g = np.zeros_like(bias_g)

        dnext_hidden = np.zeros_like(hiddens[0])
        dnext_cell = np.zeros_like(cells[0])

        loss = 0

        for t in reversed(range(len(outputs))):

            #print('i', outputs)

            loss += Utils.loss(outputs[t], targets[t])
            
            doutput_v = Utils.d_loss(outputs[t], targets[t]) #dLoss/dy_hat

            dbias_v += doutput_v * 1 #dLoss / dbv
            dweight_v += np.dot(doutput_v, hiddens[t].T) #dLoss / dwv

            dhidden = np.dot(weight_v.T, doutput_v) #dLoss/dh
            dnext_hidden += dhidden

            ### Calcul des gradients des portes des cellules LSTM ###

            dout = dhidden * Utils.tanh(cells[t])

            dbias_out += dout
            dweight_out += np.dot(dout, z_values[t].T)

            dcell = np.copy(dnext_cell)
            dcell += out_gates[t] * dhidden * Utils.d_tanh(cells[t])

            dcandidate = Utils.d_tanh(dcell * input_gates[t]) * dcell * input_gates[t]

            dweight_g += np.dot(dcandidate, z_values[t].T)
            dbias_g += dcandidate

            cell_previous = cells[t-1]

            dinput = Utils.d_sigmoid(input_gates[t]) * dcell * cell_previous
            dweight_input += np.dot(dinput, z_values[t].T)
            dbias_input += dinput

            dforget = Utils.d_sigmoid(forget_gates[t]) * dcell * cell_previous
            dweight_forget += np.dot(dforget, z_values[t].T)
            dbias_forget += dforget

            ### Calcul du de dLoss/dhidden et dLoss/dcell

            dz = (
                np.dot(dweight_forget.T, dforget) +
                np.dot(dweight_input.T, dinput) +
                np.dot(dweight_g.T, dcandidate) +
                np.dot(dweight_out.T, dout)
                )

            dh_prev = dz[:hidden_size, :]
            dh_prev = forget_gates[t] * dcell

        raw_weights_grads = dweight_input, dweight_forget, dweight_g, dweight_out, dweight_v
        raw_biases_grads = dbias_input, dbias_forget, dbias_g, dbias_out, dbias_v
        raw_grads = raw_weights_grads + raw_biases_grads

        self.grads = raw_grads
        self.clip_gradients()

        return loss

    #Peremt d'entrainer le réseau
    def train(self, datas_gen_method=Utils.gen_datas,
              n_sequences=3,
              epochs=100,
              learning_rate_method=Utils.learning_rate,
              parts=(.65, .25, .1),
              no_encode_needed = False):

        #Paramètres:
        #-inputs_method : Référence de la méthode utilisée pour générer des données à étudier
        #-n_sequences : Nombre de données à générer (plus il y en a, plus le résultat aura tendance à être bon mais plus
        # la descente du coût sera lente.
        #-epochs : Nombre de fois ou le réseau va calculer chaque entrée et modifier les paramètres
        #-learning_rate_method : Référence de la méthode utilisée pour calculer le taux d'apprentissage. On aurait aussi
        # pu donner le nombre directement, mais donner une méthode qui renvoie le nombre permet de le modifier au court du temps.
        # J'ai décidé de donner l'index de l'epoque en paramètre de la méthode de learning_rate. Cela permet de le réduire
        # au fil du temps pour se rapprocher correctement et pas trop vite du but.
        # -parts : Tuple qui contient la répartition des ensembles d'entrainement, de validation et de test.

        tstart = time.time()

        datas, dic = datas_gen_method(n_sequences)

        self.vocab_size = len(dic)//2
        self.z_size = self.vocab_size + self.hidden_size
        self.init_params()

        training_set, validation_set, testing_set = Utils.part(datas, parts)

        training_losses, validation_losses = [], []
        
        for epoch in range(1, epochs + 1):

            epoch_training_loss = 0
            epoch_validation_loss = 0

            # 1) Validation

            for inputs, targets in zip(validation_set[0], validation_set[1]):

                if not no_encode_needed:
                    inputs = Utils.sent_hot_encode(inputs, dic)
                    targets = Utils.sent_hot_encode(targets, dic)

                forward_outputs = self.forward(inputs)

                outputs = forward_outputs[-1:]

                epoch_validation_loss += LSTM.compute_loss(forward_outputs, targets)
##                epoch_validation_loss += self.backward(forward_outputs, targets)

            # 2) Training
            
            for inputs, targets in zip(training_set[0], training_set[1]):
                
                if not no_encode_needed:
                    inputs = Utils.sent_hot_encode(inputs, dic)
                    targets = Utils.sent_hot_encode(targets, dic)

                forward_outputs = self.forward(inputs)

                epoch_training_loss += self.backward(forward_outputs, targets)

                # 3) Updating

                self.update_parameters(learning_rate_method(epoch))
            training_losses.append(epoch_training_loss/len(training_set[0]))
            validation_losses.append(epoch_validation_loss/len(validation_set[0]))

            if (epoch % 100 == 0 or epoch == 1) and epoch != epochs:

                tnow=time.time()
                
                print(f"Epoque {epoch}: \nCoût (phase entraînement) : {training_losses[-1]}\nCoût" +
              f"(phase de validation) : {validation_losses[-1]}  en {tnow-tstart}\n")

                rand = np.random.randint(len(testing_set[0]))
                input_testing = testing_set[0][rand]
                target_testing = testing_set[1][rand]
        
                if not no_encode_needed:
                    input_testing = Utils.sent_hot_encode(input_testing, dic)
                    target_testing = Utils.sent_hot_encode(target_testing, dic)

                self.print_prediction(input_testing, target_testing, dic)
            if epoch % 100 == 0 or epoch == 1:
               self.save_params(epoch)

            if epoch % 100 == 0:
                thread = ShowFigure(np.arange(epoch), (training_losses, validation_losses), self.name, tnow-tstart)
                thread.run()
                    
            # 4) Répéter 1-4 jusqu'à ce qu'epoch == epochs.

        print(f"Epoque {epoch}: \nCoût (phase entraînement) : {training_losses[-1]}\nCoût" +
              f"(phase de validation) : {validation_losses[-1]} ")
        self.save_params(epochs)
    
        # 5) Testing

        rand = np.random.randint(len(testing_set))
        input_testing = testing_set[0][rand]
        target_testing = testing_set[1][rand]
        
        if not no_encode_needed:
                    input_testing = Utils.sent_hot_encode(input_testing, dic)
                    target_testing = Utils.sent_hot_encode(target_testing, dic)

        self.print_prediction(input_testing, target_testing, dic)

        # Affichage de graphe avec matplotlib.

        assert epochs == len(training_losses)
        epoch = np.arange(epochs)
        plt.figure()
        plt.plot(epoch, training_losses, 'r', label="Erreur (phase d'entraînement)")
        plt.plot(epoch, validation_losses, 'b', label="Erreur (phase de validation)")

        plt.ylim(0, 5)

        plt.legend()
        plt.xlabel("Époque"), plt.ylabel("Erreur")
        plt.show()
        
# Betises que j'ai faites:
# -Redondance de données (j'ajoutais plusieurs fois les mêmes)
# -Le param n_sequences était de base très faible.


n_sequences = 50

hidden_size = 100

    
network = LSTM(name="model_h_"+str(hidden_size), hidden_size=hidden_size)
network.train(datas_gen_method=Utils.gen_datas,
              n_sequences=n_sequences,
              epochs=1000,
              parts=(7/10,2/10,1/10),
              no_encode_needed=False)
             
