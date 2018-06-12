import retro
import time
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

from Cart import Cart

env = retro.make(game='StreetFighterII-Genesis', state='round1')
env.reset()

maxGameIterations = 2710

def runCart(cart, numGames=1, render=False):
    
    # cart = Cart(mkBrain())
    # cart.brain.set_weights(weight)

    rememberedSteps = 20

    RyuWins = 0
    ChunLiWins = 0

    currentScore = 0
    scoreTotal = 0
    while True:
        prev_infos = np.array([[0]*7]*rememberedSteps)
        prev_info = []
        gameTime = 0
        currentScore = 0
        env.reset()
        while True:
            
            gameTime+=1
            # print(gameTime)
            
            if render:
                env.render()
                time.sleep(0.01)

            
            if (gameTime < 2 + rememberedSteps):
                action = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # action = env.action_space.sample()
                # print(action)
                
            else:
                # print(prev_info)
                # print(np.shape(prev_infos.flatten()))
                # print(np.shape(prev_infos))
                a = np.reshape(prev_infos.flatten(), (1, 7*rememberedSteps))
                # print(np.shape(a))
                predict = cart.brain.predict(a)[0]
                # print(max(predict))
                if (max(predict) < 1):
                    action = np.round(predict).astype(int)
                else:
                    action = np.round(predict/max(predict)).astype(int)
                # print(action)
 
            observation, reward, done, info = env.step(action)    

            prev_info = infoToArray(info)

            if (gameTime < 1 + rememberedSteps):
                prev_infos[rememberedSteps-gameTime] = prev_info
            else:
                currentScore += evaluateRewardinGame(info, prev_infos[0])
                prev_infos = prev_infos[:-1]
                prev_infos = np.append([prev_info], prev_infos, axis=0)

            if ((info["health"] > info["enemyHealth"]) and (info["enemyHealth"] < 1)):
                # print(info)
                currentScore *= (maxGameIterations*2)/gameTime
                RyuWins += 1
                scoreTotal += currentScore
                break
            if ((info["health"] < info["enemyHealth"]) and (info["health"] < 1)):
                # print(info)
                currentScore *= (maxGameIterations*2)/gameTime
                ChunLiWins += 1
                scoreTotal += currentScore
                break
            if gameTime > maxGameIterations:
                if (info["health"] > info["enemyHealth"]):
                    RyuWins += 1
                else:
                    ChunLiWins += 1
                # print(info)
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
        rew += prev_info[1] - info["enemyHealth"]

    return rew

def mkBrain():
    
    model = Sequential()
    model.add(Dense(units= 16, activation='relu',input_dim=140))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=12, activation='relu'))

    return model

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


json_file = open('model7.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
brain = model_from_json(loaded_model_json)

brain.load_weights("model6Gen1.h5")

cart = Cart(brain)

runCart(cart, 1, True)
