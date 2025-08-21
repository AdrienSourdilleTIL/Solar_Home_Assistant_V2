Solar+Storage RL Energy Management
Overview

This project aims to develop a reinforcement learning (RL) agent to manage a residential solar + battery storage system connected to the grid. The agent’s goal is to optimize electricity usage and storage to minimize costs, maximize revenue, and smartly manage battery life.

The system can take the following actions at any time:

Charge battery from PV production

Charge battery from the grid

Discharge battery to home consumption

Discharge battery to the grid

Stay idle

The RL agent must make decisions under uncertainty, relying on forecasts and past observations to optimize outcomes in a real-world setting.

Objectives

Cost saving and revenue maximization – The primary goal is to reduce electricity costs or earn revenue by intelligently using the battery and grid.

Battery preservation – Encourage smart battery usage to extend battery life (secondary objective).

Self-sufficiency – Maintain capacity to support the home during potential grid outages (secondary objective).

Data Requirements

To train and operate the agent, the following data is collected:

Day-ahead electricity spot prices (hourly)

Historical PV production (hourly)

Weather data (temperature, solar irradiance, cloud cover, etc.)

Home consumption data (hourly)

Forecasts (optional for training, can include PV production and consumption forecasts)

The training dataset will combine all relevant features for each hour, including spot prices, production, consumption, and environmental variables.

Methodology

Data Gathering – Pull historical and forecasted spot prices, PV production, weather, and consumption data.

Data Preprocessing – Align data in an hourly timeseries, handle missing values, and optionally add noise to simulate forecast errors.

RL Environment Setup – Define state space (features), action space (possible battery/grid actions), and reward function (cost/revenue optimization with secondary objectives).

Training the Agent – Use RL algorithms (e.g., PPO, DQN, or recurrent policies) to train the agent under realistic conditions.

Evaluation – Test the agent on unseen data and analyze cost savings, battery usage, and performance under uncertainty.
