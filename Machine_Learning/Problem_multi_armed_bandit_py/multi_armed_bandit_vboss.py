import random
import math

class Bandit:

    def __init__(self, prob):
        self.prob = prob
        self.pulls = 0
        self.wins = 0

    def pull(self):
        self.pulls += 1

        chance = random.random()

        if chance < self.prob:
            self.wins += 1
            return True

        return False

n = 10000
bandits = []

b1 = Bandit( 0.2 )
b2 = Bandit( 0.3 )
b3 = Bandit( 0.7 )

bandits.append( b1 )
bandits.append( b2 )
bandits.append( b3 )

for i in range(0, n):

    best_ucb = 0
    best_bandit = None

    for bandit in bandits:
        if bandit.pulls == 0:
            best_bandit = bandit
            break
        else:
            delta = bandit.wins / bandit.pulls
            ucb = delta + math.sqrt( 2 * math.log( i ) / bandit.pulls )

            if ucb > best_ucb:
                best_ucb = ucb
                best_bandit = bandit
    
    best_bandit.pull()

for i, bandit in enumerate(bandits):
    print('Bandit: {}, wins: {}, pulls: {}, %: {}'.format( i, bandit.wins, bandit.pulls, bandit.wins / bandit.pulls ))