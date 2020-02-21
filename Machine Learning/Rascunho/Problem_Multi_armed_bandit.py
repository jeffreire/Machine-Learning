import math
import random

# teste = numpy.arange(10000)
# print(numpy.argmin(teste))

# teste = numpy.random.randint(16, size=(4, 4))
# print(teste)
# print(numpy.argmax(teste))
# bandit = 0.2

# pull = random.randint(0,1)
# print(pull)
# if pull == 0:
#     print('You loss!')

# banditResult = numpy.random.randint(bandit)
# print('You Win: {}'.format(banditResult))

# valueMin = numpy.argmin(bandit)
# print('Max: {}'.format(valueMax))
# print('Min: {}'.format(valueMin))


# [1/100]
# episo = 1000

bandits = (0.6, 0.4 , 0.8)

class Bandit:

    def __init__(self, chance):
        self.chance = chance
        print(chance)

    def pull(self):
        prob = random.random() # 0..1
        if prob < self.chance:
            return 'win'
        return 'Loss'
    
    def average(self, win, pulls):
          self.avarege_bandit = win / pulls
          return self.avarege_bandit
    
    def Score(self, routBandit):
        delta = (2 * math.log(self.routes) / routBandit)
        ucb = self.average + delta
        return ucb

    def _play(self, routes):
        
        self.routes = routes

        win1 = 0
        win2 = 0
        win3 = 0
        while routes:
            b1 = Bandit(bandits[0])
            if  b1.pull() == 'win':
                win1 += 1
            
            b2 = Bandit(bandits[1])
            if  b2.pull() == 'win':
                win2 += 1

            b3 = Bandit(bandits[2])
            if  b3.pull() == 'win':
                win3 += 1

            r = input('Deseja continuar?')
            if r != 'sim':
                break
        else:
            print(b1.average(win1, routes), 'res1: {}'.format(b1.pull()))
            print(b2.average(win2, routes), 'res2: {}'.format(b2.pull()))
            print(b3.average(win3, routes), 'res3: {}'.format(b3.pull()))
            print('Fim')
teste = Bandit(bandits)
teste._play(10)