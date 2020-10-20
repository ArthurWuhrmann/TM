from threading import Thread
import matplotlib.pyplot as plt
import os
class ShowFigure(Thread):

    def __init__(self, epoch, losses, name, time):
        Thread.__init__(self)

        self.epoch = epoch
        self.training_losses, self.validation_losses = losses
        self.name = name
        self.time = time

    def run(self):

        plt.figure()
        plt.plot(self.epoch, self.training_losses, 'r', label="Erreur (phase d'entraînement)")
        plt.plot(self.epoch, self.validation_losses, 'b', label="Erreur (phase de validation)")

        plt.legend()
        plt.xlabel("Époque"), plt.ylabel("Erreur")

        path = self.name + "/" + str(self.epoch[-1:][0] + 1)

        if not os.path.isdir(path):
            os.makedirs(path)

        delay = float(int(self.time*1e3))/1e3
        delay = str(delay)[:str(delay).find(".")+2].replace(".", "_")
        if len(delay) < 3:
            delay = "first"

        plt.ylim(0, 5)
        
        plt.savefig(self.name + "/" + str(self.epoch[-1:][0] + 1) + "/t" + str(delay) + "s")
        plt.close()

    def update(self, epoch, losses):

        self.epoch = epoch
        self.training_losses, self.validation_losses = losses
