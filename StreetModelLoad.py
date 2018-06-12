import random
import numpy as np
import time
import heapq
from keras.models import model_from_json

from statistics import median, mean
from multiprocessing import Process, Pool, Queue

import retro

from Cart import Cart


maxGameIterations = 2710
env = retro.make(game='StreetFighterII-Genesis', state='round1')
env.reset()

def getGroupOfGoodCarts(carts, quantity):
    scores=[cart.score for cart in carts]
    bestScores = heapq.nlargest(quantity, scores)
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

def infoToArray(info):
    infoArray = np.array([0]*7)

    infoArray[0] = info["health"]
    infoArray[1] = info["enemyHealth"]
    infoArray[2] = info["posx"]
    infoArray[3] = info["posxEnemy"]
    infoArray[4] = info["jumpHeight"]
    infoArray[5] = info["isAttacking"]
    infoArray[6] = info["attackIndex"]

    return infoArray
    

def mkBrain():
    from keras.models import Sequential
    from keras.layers import Dense

    rememberedSteps = 20
    
    model = Sequential()
    model.add(Dense(units= 16, activation='relu',input_dim=7*rememberedSteps))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=12, activation='relu'))

    return model



def runBestCart(cart, numGames):
    runCart(cart, numGames, True)
    
    print("Cart Scores Sum = " + str(cart.score))

def runCart(cart, numGames=1, render=False):

    rememberedSteps = 20

    RyuWins = 0
    ChunLiWins = 0

    currentScore = 0
    scoreTotal = 0
    for each_game in range(numGames):
        prev_infos = np.array([[0]*7]*rememberedSteps)
        prev_info = []
        gameTime = 0
        currentScore = 0
        env.reset()
        while True:
            
            gameTime+=1
            
            if render:
                env.render()
                # time.sleep(0.02)

            
            if (gameTime < 2 + rememberedSteps):
                action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                
            else:
                a = np.reshape(prev_infos.flatten(), (1, 7*rememberedSteps))
                predict = cart.brain.predict(a)[0]
                if (max(predict) < 1):
                    action = np.round(predict).astype(int)
                else:
                    action = np.round(predict/max(predict)).astype(int)
 
            observation, reward, done, info = env.step(action)

            prev_info = infoToArray(info)

            if (gameTime < 1 + rememberedSteps):
                prev_infos[rememberedSteps-gameTime] = prev_info
            else:
                currentScore += evaluateRewardinGame(info, prev_infos[0])
                prev_infos = prev_infos[:-1]
                prev_infos = np.append([prev_info], prev_infos, axis=0)

            if ((info["health"] > info["enemyHealth"]) and (info["enemyHealth"] < 1)):
                currentScore *= (maxGameIterations*2)/gameTime
                RyuWins += 1
                scoreTotal += currentScore
                break

            if ((info["health"] < info["enemyHealth"]) and (info["health"] < 1)):
                currentScore *= (maxGameIterations*2)/gameTime
                ChunLiWins += 1
                scoreTotal += currentScore
                break

            if gameTime > maxGameIterations:
                if (info["health"] > info["enemyHealth"]):
                    RyuWins += 1
                else:
                    ChunLiWins += 1
                scoreTotal += currentScore
                break

    cart.score = scoreTotal
    print(cart.score)

    return scoreTotal

def evaluateRewardinGame(info, prev_info):
    rew = 0

    # if lost life
    if (info["health"] < prev_info[0]):
        rew += info["health"] - prev_info[0]

    # if attacked the enemy
    if (info["enemyHealth"] < prev_info[1]):
        rew += (prev_info[1] - info["enemyHealth"])*1.5

    if (np.abs(info["posx"] - info["posxEnemy"]) > 100):
        rew += 0.1

    return rew

def saveBestfromGenModel(cart, gen):
    cart.brain.save_weights("model7Gen{}.h5".format(gen))

def saveBestToGoodCarts(cart, gen):
    cart.brain.save_weights("DeuCerto/model7Gen{}.h5".format(gen))

def overwriteAllGenModels(carts):
    for i in range(len(carts)):
        carts[i].brain.save_weights("model7Cart{}.h5".format(i))

def reproduction(survivors, childsPerCouple=1, variation=0.01, mutateProb=0.01):
    newGeneration = []
    # newGeneration += survivors
    print([s.score for s in survivors])
    for i in range(len(survivors)):
        for j in range(len(survivors)):
            if (i < j):
                print(i)
                print(j)
                newChild = crossover(survivors[i], survivors[j])
                mutate(newChild, variation, mutateProb)
                newGeneration.append(newChild)

    return newGeneration

def mutate(cart, variation=0.01, probability=1):
    getWeights = cart.brain.get_weights()
    newWeights = getWeights

    for layer in range(len(getWeights)):
        if getWeights[layer].ndim == 2:
            for i in range(len(getWeights[layer])):
                for j in range(len(getWeights[layer][i])):
                    if (random.random() < probability):
                        newWeights[layer][i][j] = random.gauss(getWeights[layer][i][j], variation)
                        
    cart.brain.set_weights(newWeights)

def crossover(cart1, cart2):
    
    maxScoreDif = 1000

    brain = mkBrain()

    getWeights1 = cart1.brain.get_weights()
    getWeights2 = cart2.brain.get_weights()

    scoreDif = cart1.score-cart2.score

    pickChance1 = (scoreDif+maxScoreDif)/(maxScoreDif*2)
    print("Chance " + str(pickChance1))

    if pickChance1 < 0:
        pickChance1 = 0
    elif pickChance1 > 1:
        pickChance1 = 1

    changed = 0
    totalTrue = 0
    total = 0

    for layer in range(len(getWeights2)):
        if getWeights2[layer].ndim == 2:
            for i in range(len(getWeights2[layer])):
                for j in range(len(getWeights2[layer][i])):
                    if (random.random() < pickChance1):
                        totalTrue += 1
                        one = getWeights2[layer][i][j]
                        getWeights2[layer][i][j] = getWeights1[layer][i][j]
                        if (one != getWeights2[layer][i][j]):
                            changed += 1
                    total += 1

    print(changed)
    print(totalTrue)
    print(total)
    print("")
    brain.set_weights(getWeights2)
    return Cart(brain)

def run(carts, numGamesPerCart, numGenerations, childsPerSurvivor, startingGen):
    bestCartFromEachGen = []
    bestChilds = carts
    genCounter = startingGen
    while True:
        print("Reproducing...")
        childs = reproduction(bestChilds, childsPerSurvivor, 0.1, 0.015)

        print("Running Childs...")

        for child in childs:
            runCart(child, numGamesPerCart)

        bestChild = getBestCart(childs)
        bestChilds = getGroupOfGoodCarts(childs, 7)
        print("\nBest from Generation " + str(genCounter+2))
        print(bestChild.score)
        print("")

        bestCartFromEachGen.append(bestChild)

        genCounter += 1
        saveBestfromGenModel(bestChild, genCounter+1)
        overwriteAllGenModels(childs)

        if (bestChild.score > 600):
            saveBestToGoodCarts(bestChild, genCounter+1)
    
    return bestChild, bestCartFromEachGen




# ----------------------------------------------------------------- #




numGamesPerCart = 1
childsPerCouple = 1

carts = []

import os.path

fileNum = 0

file = "model7Cart0.h5"

json_file = open('model7.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

while os.path.isfile(str(file)):
    brain = model_from_json(loaded_model_json)
    brain.load_weights(file)
    carts.append(Cart(brain))

    fileNum += 1
    file = "model7Cart{}.h5".format(fileNum)

fileNum = 0

file = "model7Gen0.h5"

while os.path.isfile(str(file)):
    fileNum += 1
    file = "model7Gen{}.h5".format(fileNum)

for cart in carts:
    runCart(cart)

bestChilds = getGroupOfGoodCarts(carts, 7)

TheCart, TheBestCarts = run(bestChilds, numGamesPerCart, 100, childsPerCouple, fileNum-2)




# Model 7 NN mais complexo 140, score de ataque * 1.5, crossover de 700 CONSERTADO, numGamesPerCart = 1, selecting 7, saving fixed, score when close
