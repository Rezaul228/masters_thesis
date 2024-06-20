# DQN model for multi-agent reinforcement learning 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from custom_production_env import *
import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import dqn
import matplotlib.pyplot as plt
import os
import pandas as pd


def train(episodes, num_of_orders, path, save_every = 10, show_printouts = True,
          hidden_layers_sizes=[64, 64], replay_buffer_capacity = 10000, batch_size = 32,
          step_penalty = -1, success_reward = 100):
    #Save environment model every x episodes
    env.reset(num_of_orders,show_printouts,step_penalty,success_reward)
    info_state_size = env.get_info_state_shape()  # Flattened vector from environment, pulled from env.get_info_state_shape call
    print("Info state size:",info_state_size)
    num_actions = 5                               # [do_nothing, machines 1-6], 1 + num of machines

    with tf.Session() as sess:
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                learning_rate=0.1,
                update_target_network_every=1000,
                epsilon_start=1.0,
                epsilon_end=0.1,
                learn_every= 3,
                min_buffer_size_to_learn=1000,
                epsilon_decay_duration= 5500,

                replay_buffer_capacity=replay_buffer_capacity,

                batch_size=batch_size) for idx in range(num_of_orders)
        ]
        sess.run(tf.global_variables_initializer())
        results = []
        timestep_history = []
        epsilon_history = []

        csv_columns = ["Episode/Agents"]
        for agent_counter in range(num_of_orders):
            csv_columns.append("Agent-" + str(agent_counter))
        ToCSV.initiate(csv_columns, path)

        for episode in range(episodes):
            print(f"RUNNING EPISODE {episode}")
            #TimeStep object from environment, see TimeStep Class in env
            time_step = env.reset(num_of_orders, show_printouts, step_penalty, success_reward)
            current_time = 0
            rewards = 0
            while not time_step.last():




                player_id = current_time % num_of_orders
                current_action = time_step.observations["legal_actions"][player_id]
                print ('current_action', current_action)


                if current_action[0] != 0:
                    # Extract move from DQN's agent
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                legal_actions = time_step.observations['legal_actions']
                # Perform action on environment
                time_step = env.turn_based_step(current_time, action_list, legal_actions, show_printouts)
                current_time += 1
                # if ligal action is not 0:
                rewards += sum(time_step.rewards) / len(time_step.rewards)

                epsilon = sum([agent._get_epsilon(False) for agent in agents]) / len(agents)
                epsilon_history.append((episode, current_time, epsilon))
                    # print("TimeStep Rewards",time_step.rewards)




                #
                # #...................................................................
                # if current_action[0] != 0 and player_id != 0:
                #     #Extract move from DQN's agent
                #     agent_output = agents[player_id].step(time_step)
                #     action_list = [agent_output.action]
                #     legal_actions = time_step.observations['legal_actions']
                #     # Perform action on environment
                #     time_step = env.turn_based_step(current_time, action_list, legal_actions, show_printouts)
                #     current_time += 1
                #     # if ligal action is not 0:
                #     rewards += sum(time_step.rewards) / len(time_step.rewards)
                #
                #     epsilon = sum([agent._get_epsilon(False) for agent in agents]) / len(agents)
                #     epsilon_history.append((episode, current_time, epsilon))
                #     # print("TimeStep Rewards",time_step.rewards)
                #
                # elif current_time == 0 and player_id == 0:
                #     current_time = current_time
                #     action_list = [1]
                #     legal_actions = [[1],[1,2]]
                #     show_printouts = True
                #     time_step = env.turn_based_step(current_time, action_list, legal_actions, show_printouts)
                #     current_time += 1
                #     # if ligal action is not 0:
                #     rewards += sum(time_step.rewards) / len(time_step.rewards)
                #
                #     epsilon = sum([agent._get_epsilon(False) for agent in agents]) / len(agents)
                #     epsilon_history.append((episode, current_time, epsilon))
                #
                # elif current_time == 8 and player_id == 0:
                #     current_time = current_time
                #     action_list = [3]
                #     legal_actions = [[3], [3, 4]]
                #     show_printouts = True
                #     time_step = env.turn_based_step(current_time, action_list, legal_actions, show_printouts)
                #     current_time += 1
                #     # if ligal action is not 0:
                #     rewards += sum(time_step.rewards) / len(time_step.rewards)
                #
                #     epsilon = sum([agent._get_epsilon(False) for agent in agents]) / len(agents)
                #     epsilon_history.append((episode, current_time, epsilon))
                # else:
                #     time_step = env.turn_based_step(current_time, action_list, legal_actions, show_printouts)
                #     current_time += 1
                #     # if ligal action is not 0:
                #     rewards += sum(time_step.rewards) / len(time_step.rewards)
                #
                #     epsilon = sum([agent._get_epsilon(False) for agent in agents]) / len(agents)
                #     epsilon_history.append((episode, current_time, epsilon))
                    #agent_output includes probabilities. Only pull action
                    #......................................................

                #print("Rewards:",rewards)
                #print("info_state_size",env.get_info_state_shape())

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
                #print(agent.get_weights())

            #GenerateSimData.generate_task_status(cls.all_agents)
            time_it_took = current_time//num_of_orders
            print(f"It took {time_it_took} unit of time to complete {SimParams.N_ORDER} agents !! Bye Bye :) !!")
            results.append(rewards)
            timestep_history.append(time_it_took)

            if episode%save_every == 0:
                for agent in agents:
                    agent.save(path+f"weights/{episode}/")
                    # print(agent.get_weights())
                plt.plot(results)
                plt.title("Rewards per Episode")
                plt.xlabel("Episode")
                plt.ylabel("Rewards")
                plt.savefig(path+f"plots/{episode}.png")
                with open(path+f"history.txt", "w") as f:
                    for i in range(len(results)):
                        f.write(str(results[i]) + ", " + str(timestep_history[i]) +"\n")
                env.write_to_file(path+f'/simdata_train/sim_data_{episode}.json')
                df_results = pd.DataFrame({'Rewards': results, 'TimeSteps': timestep_history})
                df_results.to_csv(path+f'/simdata_train/rewards.csv')
                df_epsilon = pd.DataFrame(epsilon_history,columns=['Episode','TimeStep','Epsilon'])
                df_epsilon.to_csv(path+'epsilon.csv')

                print("SAVED!")

            # end of an episode, write the agents job completion history to csv
            env.exp_episode_to_csv(episode)


        saver = tf.train.Saver()

def test(episodes, num_of_orders, path, show_printouts = True, load_weights_from_episode = 0,
         hidden_layers_sizes=[64, 64], replay_buffer_capacity=10000, batch_size=32):
    env.reset(num_of_orders,show_printouts)
    info_state_size = env.get_info_state_shape()  # Flattened vector from environment, pulled from env.get_info_state_shape call
    print("Info state size:",info_state_size)
    num_actions = 5                               # [do_nothing, machines 1-6]


    with tf.Session() as sess:
        #Set up agents in DQN, correspond to agent ID in environment
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=replay_buffer_capacity,
                batch_size=batch_size) for idx in range(num_of_orders)
        ]
        sess.run(tf.global_variables_initializer())
        for agent in agents:
             agent.restore(path+f"weights/{load_weights_from_episode}/")
        results = []
        timestep_history = []
        for episode in range(episodes):
            print(f"RUNNING EPISODE {episode}")
            #TimeStep object from environment, see TimeStep Class in env
            time_step = env.reset(num_of_orders, show_printouts)
            rewards = 0
            current_time = 0
            while not time_step.last():
                player_id = current_time % num_of_orders
                #current_action = time_step.observations["legal_actions"][player_id]
                #Extract move from DQN's agent
                #if current_action[0] != 0:
                agent_output = agents[player_id].step(time_step, is_evaluation=True)
                    #agent_output includes probabilities. Only pull action
                action_list = [agent_output.action]
                # Perform action on environment
                time_step = env.turn_based_step(current_time, action_list, show_printouts)

                current_time += 1
                #rewards+= time_step['rewards']
                rewards += sum(time_step.rewards) / len(time_step.rewards)

            env.write_to_file(path+f'/simdata_test/sim_data_{episode}.json')
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step, is_evaluation=True)
            #GenerateSimData.generate_task_status(cls.all_agents)
            time_it_took = current_time//num_of_orders
            results.append(rewards)
            timestep_history.append(time_it_took)
            df_results = pd.DataFrame({'Rewards': results, 'TimeSteps':timestep_history})
            df_results.to_csv(path + f'/simdata_test/rewards.csv')
            print(f"It took {time_it_took} unit of time to complete {SimParams.N_ORDER} agents !! Bye Bye :) !!")
    return  timestep_history

#For comparison
def run_old(episodes = 1, num_of_orders=0, show_printouts = False):
    GenerateSimData.generate_machine_info()
    results = []
    for episode in range(episodes):
        print(f"RUNNING EPISODE {episode}")
        env.reset_old()
        time_it_took = env.run(num_of_orders, show_printouts)
        results.append(time_it_took)
    return results

if __name__ == "__main__":
    TRAIN =True
    #TRAIN = False
    #Run for DQN


    # Where to save/load models to/from
    path = "/home/rezaul/Developments/open_spiel/open_spiel/python/environments/train/train/"

    # for containerdocker
    #path = "/repo/open_spiel/python/examples/train/"

    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+'plots/'):
        os.mkdir(path+'plots/')
    if not os.path.exists(path+'weights/'):
        os.mkdir(path+'weights/')
    if not os.path.exists(path+'simdata_test/'):
        os.mkdir(path+'simdata_test/')
    if not os.path.exists(path+'simdata_train/'):
        os.mkdir(path+'simdata_train/')#


    # Save models every x episodes.
    show_printouts = True
    train_episodes = 10001
    save_every = 100
    test_episodes = 500
    load_weights_from_episode = 10000
    # TODO: Change, I changed it to 2 as we have 2 fixed agents
    num_of_orders = 2
    step_penalty = -1
    success_reward = 100


    # DQN model hyper-parameters
    hidden_layers_sizes = [128,128]     #size of the hidden neural network layers in the DQN
    replay_buffer_capacity = 10000 #int(1e6)
    batch_size = 64

    if TRAIN:
        print("TRAINING!")
        train(episodes = train_episodes, num_of_orders = num_of_orders,
              path = path, save_every = save_every, show_printouts = show_printouts,
              hidden_layers_sizes = hidden_layers_sizes, replay_buffer_capacity = replay_buffer_capacity,
              batch_size=batch_size, step_penalty = step_penalty, success_reward = success_reward)
    else:
        print("TESTING")
        print("DQN:\n\n")
        dqn_results = test(episodes = test_episodes, num_of_orders=num_of_orders,
                           path=path, show_printouts=show_printouts, load_weights_from_episode = load_weights_from_episode,
                           hidden_layers_sizes=hidden_layers_sizes, replay_buffer_capacity=replay_buffer_capacity,
                           batch_size=batch_size)
        print("HARD CODED:\n\n")
        hard_coded_results = run_old(episodes = test_episodes, num_of_orders = num_of_orders)
        print(f"DQN AVERAGE: {sum(dqn_results)/len(dqn_results)}")
        print(f"HARD CODED AVERAGE: {sum(hard_coded_results)/len(hard_coded_results)}")

    with open(path + 'finish.md', 'w') as fp:
        fp.write('DONE')
    fp.close()
