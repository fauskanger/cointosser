import os
import random
import itertools
from collections import Counter
import math


class Coin(object):
    head = 0
    tail = 1

    def __init__(self, face):
        self.face = face

    def __eq__(self, other):
        return self.face == other.face

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'Head' if self.face is Coin.head else 'Tail'

    @staticmethod
    def create_head():
        return Coin(Coin.head)

    @staticmethod
    def create_tail():
        return Coin(Coin.tail)

    @staticmethod
    def create_opposite_of(face):
        if face is Coin.head:
            return Coin.create_tail()
        return Coin.create_head()

    @staticmethod
    def create_random():
        return Coin(random.choice([Coin.head, Coin.tail]))

    def create_opposite(self):
        return Coin.create_opposite_of(self.face)

    def create_same(self):
        return Coin(self.face)

    def copy(self, copy_face=True):
        return self.create_same() if copy_face else self.create_opposite()

    def change(self):
        self.face = Coin.head if self.face is Coin.tail else Coin.tail


class Prediction(object):
    def __init__(self, faces=None, coins=None):
        if __debug__:
            assert faces or coins, 'Empty prediction'
        if faces:
            self.sequence = [Coin(face) for face in faces]
        elif coins:
            self.sequence = coins

    def __repr__(self):
        return '{}'.format(self.sequence)

    def imitate(self, inverts=None):
        inverts = [] if not inverts else inverts
        return Prediction(coins=[coin.copy(copy_face=i not in inverts) for (i, coin) in enumerate(self.sequence)])

    def fits(self, sequence):
        return self.sequence == sequence

    def append(self, coin):
        self.sequence.append(coin)

    def opposite(self):
        return Prediction(coins=[coin.create_opposite() for coin in self.sequence])

    def __eq__(self, other):
        return self.sequence == other.sequence

    def __ne__(self, other):
        return not self == other


class Experiment(object):
    start = 'start'
    second = 'second'

    def __init__(self, n_coins_per_prediction, start_prediction):
        self.prediction_length = n_coins_per_prediction
        self._winners = dict()
        self.start_prediction = start_prediction

        start_coins = start_prediction.sequence
        second_coins = [start_coins[1].create_opposite()]
        second_coins.extend([coin.create_same() for coin in start_coins[0:2]])

        self.second_prediction = Prediction(coins=second_coins)

    def update_winner(self, winner):

        try:
            self._winners[winner] += 1
        except KeyError:
            self._winners[winner] = 1

    def n_start_wins(self):
        return self._winners[Experiment.start]

    def n_second_wins(self):
        return self._winners[Experiment.second]

    def start_win_ratio(self):
        return self.n_start_wins()/self.n_results()

    def second_win_ratio(self):
        return self.n_second_wins()/self.n_results()

    def n_results(self):
        return self.n_start_wins() + self.n_second_wins()

    def standard_deviation_total(self):
        # p = 0.5 => p(1-p) = 0.5 * 0.5 = 0.25
        return math.sqrt(self.n_results()*0.25)

    def standard_deviation_start(self):
        return math.sqrt(self.n_results()*0.25)

    def standard_deviation_second(self):
        return math.sqrt(self.n_results()*0.25)

    def toss_sequence(self, max_toss=10000):
        predictions = {
            Experiment.start: self.start_prediction,
            Experiment.second: self.second_prediction
        }
        sequence = []
        for count in range(max_toss):
            sequence.append(Coin.create_random())
            if len(sequence) < self.prediction_length:
                continue
            for key, prediction in predictions.items():
                last_3 = sequence[-self.prediction_length:]
                if prediction.fits(last_3):
                    return key

    def run_experiment(self, n_iterations):
        for _ in range(n_iterations):
            winner = self.toss_sequence()
            self.update_winner(winner)

    def print_results(self):
        print('{}With {} coins per prediction for {} vs {}:'
              .format(os.linesep, self.prediction_length, self.start_prediction, self.second_prediction))
        print(self._winners)


def plot_ratios(experiments, n_res):
    # Adapted from example: http://matplotlib.org/examples/pylab_examples/bar_stacked.html
    import numpy as np
    import matplotlib.pyplot as plt

    n_experiments = len(experiments)
    start_means = tuple([experiment.start_win_ratio()*100 for experiment in experiments])
    second_means = tuple([experiment.second_win_ratio()*100 for experiment in experiments])
    # start_std = tuple([experiment.standard_deviation_start() for experiment in experiments])
    # second_std = tuple([experiment.standard_deviation_second() for experiment in experiments])

    ind = np.arange(n_experiments)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    win_steps = 5

    p1 = plt.bar(ind, start_means, width, color='r')
    p2 = plt.bar(ind, second_means, width, color='y', bottom=start_means)

    x_labels = tuple(['{} \n vs \n{}'
                     .format(experiment.start_prediction, experiment.second_prediction) for experiment in experiments])

    plt.ylabel('Percent of wins')
    plt.title('Percent of wins by start prediction and winner. n={}'.format(n_res))
    plt.xticks(ind+width/2., x_labels)
    plt.yticks(np.arange(0, 101, win_steps))
    plt.legend((p1[0], p2[0]), ('Start', 'Second'))

    plt.draw()
    plt.show()
    plt.savefig("myfig1{}.png".format(n_res))


def plot_counts(experiments, n_res):
    # Adapted from example: http://matplotlib.org/examples/pylab_examples/bar_stacked.html
    import numpy as np
    import matplotlib.pyplot as plt

    n_experiments = len(experiments)
    start_means = tuple([experiment.n_start_wins() for experiment in experiments])
    second_means = tuple([experiment.n_second_wins() for experiment in experiments])
    start_std = tuple([experiment.standard_deviation_start() for experiment in experiments])
    second_std = tuple([experiment.standard_deviation_second() for experiment in experiments])

    ind = np.arange(n_experiments)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    max_count = max([experiment.n_results() for experiment in experiments])+1
    win_steps = max_count / 10  # math.ceil(max_count/(0.1 * max_count)) * 0.1 * max_count

    p1 = plt.bar(ind, start_means,   width, color='r', yerr=second_std)
    p2 = plt.bar(ind, second_means, width, color='y',
                 bottom=start_means, yerr=start_std)

    x_labels = tuple(['{} \n vs \n{}'
                     .format(experiment.start_prediction, experiment.start_prediction) for experiment in experiments])

    plt.ylabel('Number of wins')
    plt.title('Number of wins by start prediction and winner. n={}'.format(n_res))
    plt.xticks(ind+width/2., x_labels)
    plt.yticks(np.arange(0, n_res+1, n_res/10))
    plt.legend((p1[0], p2[0]), ('Start', 'Second'))

    plt.draw()
    plt.show()
    plt.savefig("myfig2{}.png".format(n_res), pad_inches=0.4)


def main():
    n_res = 1000000
    experiments = []
    for n_coins in range(3, 4):
        face_permutations = itertools.product([Coin.head, Coin.tail], repeat=n_coins)
        prediction_permutations = [Prediction(faces) for faces in face_permutations]
        combinations = []
        for prediction in prediction_permutations:
            if prediction.opposite() not in combinations:
                combinations.append(prediction)
        for prediction in combinations:
            experiment = Experiment(n_coins_per_prediction=n_coins, start_prediction=prediction)
            experiments.append(experiment)
            experiment.run_experiment(n_iterations=n_res)
            experiment.print_results()

    plot_ratios(experiments, n_res)
    plot_counts(experiments, n_res)

if __name__ == '__main__':
    main()
