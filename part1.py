import csv
import numpy as np
from scipy.stats import geom
import math
import matplotlib.pyplot as plt

# part 1

def exponential_weights(payoff_matrix_p1, epsilon, h, r):
    mask = payoff_matrix_p1[0]
    no_mask = payoff_matrix_p1[1]
    stay_home = payoff_matrix_p1[2]

    # calculate the probability of chosing every action in round r
    probabilities, round_actions = get_probabilities(r, epsilon, h, payoff_matrix_p1)

    # chose action for round r with probabilities
    action = np.random.choice(round_actions, p=probabilities)

    # regret is 2 * h sqrt(ln(k) / h)
    # learning rate is sqrt(ln(k) / h)

    return action


def get_probabilities(r, e, h, payoff_matrix):
    hindsight_payoffs = []
    total_payoff = 0
    probabilities = []
    curr_action = []

    if r == 0:
        for action in range(len(payoff_matrix)):
            # print('action index', action)
            # action_payoff = payoff_matrix[action][0]
            curr_action.append(action)
            hindsight_payoff = 0
            hindsight_payoffs.append(hindsight_payoff)
        return [(1/3), (1/3), (1/3)], curr_action
    else:
        for action in range(len(payoff_matrix)):
            # action_payoff = payoff_matrix[action][r]
            curr_action.append(action)
            hindsight_payoff = sum(payoff_matrix[action][:r])
            hindsight_payoffs.append((1+e) ** (hindsight_payoff/h))
        total_payoff = sum(hindsight_payoffs)

        for action in range(len(payoff_matrix)):
            probabilities.append(hindsight_payoffs[action]/total_payoff)

        return probabilities, curr_action


def follow_perturbed_leader(payoff_matrix_p2, epsilon, r, hallucinations):
    # get values for each action at every round
    mask = payoff_matrix_p2[0]
    no_mask = payoff_matrix_p2[1]
    stay_home = payoff_matrix_p2[2]

    # Choose the payoff of the BIH action - take into account the round 0 hallucinations

    bih1, bih2, bih3 = best_in_hindsight(mask, no_mask, stay_home, r, hallucinations)
    max_action = np.argmax([bih1, bih2, bih3])

    return max_action


def best_in_hindsight(mask, no_mask, stay_home, curr_round, hallucinations):
    # best in hindsight DOES NOT include the current round
    # to find BIH of entire action, all it on curr_round = length of list

    bih1 = sum(mask) + hallucinations[0]
    bih2 = sum(no_mask) + hallucinations[1]
    bih3 = sum(stay_home) + hallucinations[2]

    return bih1, bih2, bih3


def theo_opt_epsilon(k, n):
    epsilon = math.sqrt(np.log(k)/n)

    return epsilon

if __name__ == "__main__":
    # bimatrix game 1
        # health: (-5, 0, 5)
        # grocery: 2
        # comfort: 1
    bimatrix_payoffs_1 = [[[7,7], [2,3], [7,6]],
                            [[3,2], [-2,-2],[8,6]],
                            [[6,7], [6,8], [6,6]]]
    h1 = 8

    combined_action = [[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]
    
    # payoff matrix for player 1 - sent into EW                          
    payoff_matrix_p1 = {
        0: [],
        1: [],
        2: []
    }
    
    # payoff matrix for player 2 - sent into FTPL                          
    payoff_matrix_p2 = {
        0: [],
        1: [],
        2: []
    }

    # action array for player 1 - selected by EW                          
    action_array_p1 = []
    
    # payoff matrix for player 2 - selected by FTPL                          
    action_array_p2 = []

    # calculate learning rate
    k = 3 
    n = 100
    epsilon = theo_opt_epsilon(k, n)
    
    # generate hallucinations
    hallucinations = geom.rvs(epsilon, size=len(payoff_matrix_p2))

    possible_actions = np.array([0, 1, 2])

    p1_total_payoff = 0
    p2_total_payoff = 0
    
    combined_actions_over_time = []

    for r in range(n):
        ew_action = exponential_weights(payoff_matrix_p1, epsilon, h1, r)
        action_array_p1.append(ew_action)

        ftpl_action = follow_perturbed_leader(payoff_matrix_p2, epsilon, r, hallucinations)
        action_array_p2.append(ftpl_action)

        # Calculate payoffs from chosen action
        action_payoffs = bimatrix_payoffs_1[ew_action][ftpl_action]
        combined_actions_over_time.append(combined_action[ew_action][ftpl_action])

        # Update the payoff matrices with payoffs from actions taken
        payoff_matrix_p1[ew_action].append(action_payoffs[0])
        payoff_matrix_p2[ftpl_action].append(action_payoffs[1])

        # Update total payoff with payoff of action taken
        p1_total_payoff += action_payoffs[0]
        p2_total_payoff += action_payoffs[1]

        # fill in payoff matrix for player 1's other possible choices
        for a in np.where(possible_actions != ew_action)[0]:
            payoff_matrix_p1[a].append(bimatrix_payoffs_1[a][ftpl_action][0])
        
        # fill in payoff matrix for player 2's other possible choices
        for a in np.where(possible_actions != ftpl_action)[0]:
            payoff_matrix_p2[a].append(bimatrix_payoffs_1[a][ew_action][1])

    print(payoff_matrix_p1, payoff_matrix_p2)

    print('ew player total payoff', p1_total_payoff)
    print('ftpl player total payoff', p2_total_payoff)
    print('ew action array', action_array_p1)
    print('ftpl action array', action_array_p2)

    fig = plt.figure(1)
    plt.plot(np.arange(n), action_array_p1, label="EW")
    plt.plot(np.arange(n), action_array_p2, label="FTPL")
    plt.legend(loc='lower right')
    plt.title("Action Choices Over Time EW vs. FTPL")
    plt.ylabel("Action")
    plt.xlabel("Round Number")
    ax = fig.add_subplot(1, 1, 1)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    print(labels)
    labels[1] = 'Mask'
    labels[5] = 'No Mask'
    labels[9] = 'Stay Home'
    ax.set_yticklabels(labels)
    plt.tight_layout()
    plt.savefig('ActionChoices_n' + str(n) + '.png')

    fig = plt.figure(2)
    plt.plot(np.arange(n), combined_actions_over_time)
    plt.title("Bimatrix Actions Over Time")
    plt.ylabel("Bimatrix Action")
    plt.xlabel("Round Number")
    labels = [0,1,2,3,4,5,6,7,8]
    print(labels)
    labels[0] = '(Mask, Mask)'
    labels[1] = '(Mask, No Mask)'
    labels[2] = '(Mask, Stay)'
    labels[3] = '(No Mask, Mask)'
    labels[4] = '(No Mask, No Mask)'
    labels[5] = '(No Mask, Stay)'
    labels[6] = '(Stay, Mask)'
    labels[7] = '(Stay, No Mask)'
    labels[8] = '(Stay, Stay)'
    plt.yticks([0,1,2,3,4,5,6,7,8], labels)
    plt.tight_layout()
    plt.savefig('BiMatrixChoices_n' + str(n) + '.png')


    # CALCULATE REGRET?
    # ew_regret = calculate_regret(payoff_matrix_p1, ew)
    # ftpl_regret = calculate_regret(payoff_matrix_p2, ftpl)