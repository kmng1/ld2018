# Reinforcement Learning Demo 

## Installation
The project uses virtualenv and pip. Required software are in requirements.txt file. Installation steps

Clone this repo and enter the directory

```
git clone https://github.com/chukmunnlee/ld2018.git
cd ld2018
```

Create a virtual environment and activate it. Use Python 3.5 and greater

```
virtualenv -p /usr/bin/python3 venv
source venv/bin/activate
```

Install all the required modules

```
pip install -U -r requirements.txt
```

Should be quite painless. The only issue you might face is the openai libraries. Please go to [openai](https://github.com/openai/gym) to resolve your issues

## Running the demos

### Blackjack

There are 2 algorithm: Monte Carlo and Sarsa(λ); the files are mc.py and sarsa_lambda.py. To train either one of them over 100000 episode for example run the following command

```
python mc.py 100000
```

or 

```
python sarsa_lambda.py 100000
```

Once the training completes, the program will write out the Q-Values into the data directory. mc.py will produce a file call montecarlo_100000.pickle file; sarsa_lambda.py will produce a file called sarsa-lambda_100000.pickle.

To play Blackjack using the train Q-values run play.py file as below

```
python play.py montecarlo 100000
```

where the agent will use the Q-Values from montecarlo_100000.pickle file.

To visualize the decision run one of the following

```
python visualize.py montecarlo 100000
python viz_decison.py montecarlo 100000
```

### Cart Pole

Cart pole demo uses function approximation instead of DP (dynamic programming). To run 100 episodes of cart pole run the following

```
python cartpole_sarsa.py 100
```

The cart pole uses [sarsa with linear approximation](http://artint.info/2e/html/ArtInt2e.Ch12.S9.SS1.html)
