import random
import numpy as np
import time
import heapq

from statistics import median, mean
from multiprocessing import Process, Pool, Queue

import gym

from Cart import Cart


initialGames = 10
maxGameIterations = 500
training_data = []
acceptedScores = []
env = gym.make('CartPole-v0')
env.reset()

def job(cart, els=None):
    return 2

def init_population(numGamesPerCart):
    
    carts = [Cart(mkBrain()) for i in range(initialGames)]

    runGamesInParallel(carts, numGamesPerCart)

    print("done")

    for cart in carts:
        print(cart.score)

    bestCarts = getFiveGoodCarts(carts)
    
    return bestCarts

def getFiveGoodCarts(carts):
    scores=[cart.score for cart in carts]
    bestScores = heapq.nlargest(5, scores)
    bestCarts = [carts[scores.index(i)] for i in bestScores]
    # print(bestScores)
    # print([cart.score for cart in bestCarts])
    print("Selecting the best 5 carts...")

    return bestCarts

def getBestCart(carts):
    scores=[cart.score for cart in carts]
    bestCart = carts[scores.index(max(scores))]
    # print(max(scores))
    # print(bestCart.score)
    print("Selecting the best cart...")
    return bestCart

def mkBrain():
    from keras.models import Sequential
    from keras.layers import Dense
    
    model = Sequential()
    model.add(Dense(units= 8, activation='relu', input_dim=4))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    return model



def runBestCart(cart, numGames):
    runCart(cart, numGames, True)
    
    print("Cart Scores Sum = " + str(cart.score))

def runCart(cart, queue, numGames=1, render=False):

    # cart = Cart(mkBrain())
    # cart.brain.set_weights(weight)

    scoreTotal = 0
    for each_game in range(numGames):
        prev_obs = []
        env.reset()
        for i in range(500):
            
            if render:
                env.render()
                time.sleep(0.02)

            
            if len(prev_obs)==0:
                action = random.randrange(0,2)
                
            else:
                # print(prev_obs)
                action = np.argmax(cart.brain.predict(prev_obs))

            observation, reward, done, info = env.step(action)
            prev_obs = np.reshape(observation, (1,4))
            scoreTotal+=reward
            if done: 
                
                break

    #         print(i)
    # print("cabo o for")
    cart.score = scoreTotal
    print(cart.score)

    queue.put(cart.score)

    return scoreTotal

def reproduction(survivors, childsPerSurvivor, variation):
    newGeneration = []
    # newGeneration += survivors
    for cart in survivors:
        brain = cart.brain
        getWeights = brain.get_weights()
        newWeights = getWeights
        for _ in range(childsPerSurvivor):
            
            for layer in range(len(getWeights)):
                if getWeights[layer].ndim == 2:
                    # print(getWeights[layer])
                    for i in range(len(getWeights[layer])):
                        for j in range(len(getWeights[layer][i])):
                            newWeights[layer][i][j] = random.gauss(getWeights[layer][i][j], variation)
                    # print(newWeights[layer])
            
            brain.set_weights(newWeights)
            newGeneration.append(Cart(brain))
    
    return newGeneration

def mutate():
    pass

def run(carts, numGenerations):
    bestCartFromEachGen = []
    bestChilds = carts
    for i in range(numGenerations):
        print("Reproducing...")
        childs = reproduction(bestChilds, 3, 0.03)

        print("Running Childs...")

        runGamesInParallel(childs, numGamesPerCart)

        bestChild = getBestCart(childs)
        bestChilds = getFiveGoodCarts(childs)
        print("\nBest from Generation " + str(i+2))
        print(bestChild.score)
        print("")

        bestCartFromEachGen.append(bestChild)
    
    return bestChild, bestCartFromEachGen

def runGamesInParallel(carts, numGamesPerCart):
    
    queue = Queue()
    proc = []
    for i in range(len(carts)):
        p = Process(target=runCart, args = (carts[i], queue, numGamesPerCart))
        p.start()
        proc.append(p)
        print("Started")
    for p in proc:
        print("Joined")
        p.join()

    # for i in range(len(carts)):
    #     carts[i].score = queue.get()



numGamesPerCart = 50

# bests = []
# print("Creating First Population...")

bests = init_population(numGamesPerCart)

# bests = init_population(numGamesPerCart)

# best = getBestCart(bests)


# print("\nBest from Generation 1")
# print(best.score)
# print("")


# TheCart, TheBestCarts = run(bests, 2)

# # runBestCart(best, 5)
# # for c in TheBestCarts:
# #     runBestCart(c, 5)

# # import os.path

# # file = 0

# # while os.path.isfile(str(file)):
# #     file+=1

# # best.brain.save(str(file))


# model_json = best.brain.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

