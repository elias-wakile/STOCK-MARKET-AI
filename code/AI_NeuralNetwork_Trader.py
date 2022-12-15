import datetime
import numpy as np
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
from tqdm import tqdm


def reward_graph(model_name, ax, ay, interval):
    """
    Plot graph of reward of the model on given model
    """
    plt.plot(ax, ay)
    plt.title(f'Reward of the model with interval {interval} ' + model_name + f" as depend in time")
    plt.savefig(model_name + f"_reward_graph_{interval}.png")
    plt.ylabel("Reward in dollars")
    plt.xlabel("Iteration")
    plt.show()
    plt.close()


def reward_pre_stock_graph(model_name, xaxis, stock_reward, stock_names, interval):
    """
    Plot graph of reward per stock of the model on given model
    """
    fig, ax = plt.subplots(len(stock_names), sharex="col")
    title = f'Reward per stock with interval {interval} ' + model_name + f" as depend in time"
    save_name = model_name + f"_reward_per_stock_progress_{interval}.png"
    for ind, name in enumerate(stock_names):
        ax[ind].plot(xaxis, stock_reward[name], label=name)
        ax[ind].legend(loc=0)
    fig.suptitle(title)
    fig.supxlabel("Iteration")
    fig.supylabel("Reward in dollars")
    fig.savefig(save_name)
    fig.show()
    plt.close()


def balance_graph(model_name, ax, ay, interval):
    """
    Plot graph of balance as depend of time of a given model
    """
    plt.plot(ax, ay)
    title = f'Balance of the model with interval {interval} ' + model_name + f" as depend of time"
    save_name = model_name + f"_balance_progress_{interval}.png"
    plt.title(title)
    plt.ylabel("Money in dollars")
    plt.xlabel("Iteration")
    plt.savefig(save_name)
    plt.show()
    plt.close()


def balance_graph_together(ax, ay_net, ay_ex, interval):
    """
    Plot graph of balance as depend of time of the two model
    """
    plt.plot(ax, ay_net, label="neuralNet")
    plt.plot(ax, ay_ex, label="Extrapolation")
    title = f'Balance of the models with interval {interval} ' + f" as depend in time"
    save_name = f"balance_progress_two_models_{interval}.png"
    plt.legend(loc=0)
    plt.title(title)
    plt.ylabel("Money in dollars")
    plt.xlabel("Iteration")
    plt.savefig(save_name)
    plt.show()
    plt.close()


def run_trader(neuralNet, portfolio_agent, batch_size, stock_names, file, initial_balance, interval):
    """
    run the neural Net
    :param neuralNet: neural network that will be the model
    :param portfolio_agent: object of the portfolio
    :param batch_size: the size of the batch to retrain the model
    :param stock_names: the name of the stock that the model will work on
    :param file: the name of the file to write the results
    :param initial_balance:  the initial balance of the model
    :param interval: str of the interval of the data for plot in name graphs
    :return: list of the balance of the model in the run
    """
    i = 0
    done = False
    states = portfolio_agent.get_state().tolist()
    tmp = portfolio_agent.stock_market[portfolio_agent.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_reward = {name: [] for name in stock_names}
    balance_difference_lst = []
    current_balance = initial_balance
    balance_difference = 0
    balance_progress = []
    action_limit = int(neuralNet.action_space / 2)
    for t in tqdm(range(data_samples)):
        print(
            f'The date: {portfolio_agent.stock_market[portfolio_agent.stock_name_list[0]].stock_data["DATE"].iloc[t]}')
        file.write(
            f'The date: {portfolio_agent.stock_market[portfolio_agent.stock_name_list[0]].stock_data["DATE"].iloc[t]}'
            + '\n')
        action = []
        action_dic = {i: [] for i in range(-action_limit, action_limit + 1)}
        for ind, name in enumerate(stock_names):
            a = neuralNet.action([states[ind]]) - int(neuralNet.action_space / 2)  # make this to be between -X to X
            action.append(a)
            action_dic[a].append(ind)
        portfolio_agent.update_portfolio()
        next_states = portfolio_agent.get_state().tolist()
        results = portfolio_agent.action(action_dic)
        for ind, name in enumerate(stock_names):
            neuralNet.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
            if results[ind] > 0:
                stock_reward[name].append(results[ind])
            else:
                stock_reward[name].append(results[ind])
        states = next_states
        balance_progress.append(current_balance)
        current_balance = portfolio_agent.getBalance()
        balance_difference = portfolio_agent.profit - balance_difference
        balance_difference_lst.append(balance_difference)
        balance_difference = portfolio_agent.profit

        if len(neuralNet.memory) > batch_size:
            neuralNet.batch_train(batch_size)
        i += 1
        if t == data_samples - 1:
            done = True
    xaxis = np.array([i for i in range(data_samples)])
    y1axis = np.array(balance_progress)
    balance_graph("neuralNet", xaxis, y1axis, interval)
    yaxis = np.array(balance_difference_lst)
    reward_graph("neuralNet", xaxis, yaxis, interval)
    if len(stock_names) > 1:
        reward_pre_stock_graph("neuralNet", xaxis, stock_reward, stock_names, interval)
    return y1axis, xaxis


def run_trader_extrapolation(portfolio, file, initial_balance, stock_names, interval):
    """
    run the extrapolation
    :param portfolio: object of the portfolio
    :param file: the name of the file to write the results
    :param initial_balance:  the initial balance of the model
    :param stock_names: the name of the stock that the model will work on
    :param interval: str of the interval of the data for plot in name graphs
    :return: list of the balance of the model in the run
    """
    tmp = portfolio.stock_market[portfolio.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    balance_difference_lst = []
    current_balance = initial_balance
    stock_reward = {name: [] for name in stock_names}
    balance_progress = []
    balance_difference = 0
    states_buy = [[] for i in stock_names]
    states_sell = [[] for i in stock_names]
    for _ in tqdm(range(data_samples)):
        print(
            f'The date: {portfolio.stock_market[portfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}')
        file.write(
            f'The date: {portfolio.stock_market[portfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}' + '\n')
        actions, reward = portfolio.linear_reward()
        for ind, name in enumerate(stock_names):
            a = actions[ind]
            if a > 0:
                states_buy[ind].append(portfolio.stock_market[name].time_stamp)
            elif a < 0:
                states_sell[ind].append(portfolio.stock_market[name].time_stamp)
            stock_reward[name].append(reward[ind])
        portfolio.update_portfolio()
        balance_progress.append(current_balance)
        current_balance = portfolio.getBalance()
        balance_difference = portfolio.profit - balance_difference
        balance_difference_lst.append(balance_difference)
        balance_difference = portfolio.profit

    xaxis = np.array([i for i in range(data_samples)])
    yaxis = np.array(balance_difference_lst)
    reward_graph("Extrapolation", xaxis, yaxis, interval)
    y1axis = np.array(balance_progress)
    balance_graph("Extrapolation", xaxis, y1axis, interval)
    portfolio.getBalance()
    if len(stock_names) > 1:
        reward_pre_stock_graph("Extrapolation", xaxis, stock_reward, stock_names, interval)
    return y1axis


def main_def(start_date, end_date, stock_names):
    """
    run the extrapolation and the neural net
    :param start_date: when to start to invest
    :param end_date: when to finish to invest
    :param stock_names: the name of the stock that the model will work on
    :return: None
    """
    # vars for PortFolio
    if (end_date - start_date).days <= 14:
        initial_investment = 5000
    else:
        initial_investment = 10000
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    interval = "30m"
    stock_indices = {name: i for name, i in enumerate(stock_names)}

    episodes = 5
    # vars for NeuralNetwork
    state_size = 7
    action_space = 11
    batch_size: int = 8

    with open("result.txt", 'w') as f:
        neural_net = NeuralNetwork(episodes=episodes, state_size=state_size, action_space=action_space
                                   , model_to_load="Net.h5")

        print("Test NeuralNetwork")
        f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        f.write("Test NeuralNetwork" + '\n')
        f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        portfolio = PortFolio(initial_investment, stock_names, interval, start_date, end_date, stock_indices, f,
                              action_space)
        ay_net, ax = run_trader(neural_net, portfolio, batch_size, stock_names, f, initial_investment, interval)
        print("Test Linear")
        f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        f.write("Test Linear" + '\n')
        f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
        portfolio = PortFolio(initial_investment, stock_names, interval, start_date, end_date, stock_indices, f,
                              action_space)
        ay_extrapol = run_trader_extrapolation(portfolio, f, initial_investment, stock_names, interval)
        balance_graph_together(ax, ay_net, ay_extrapol, interval)


