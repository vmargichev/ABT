from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import random
import matplotlib.pyplot as plt

T = 3
R = 2
P = 1
S = 0

class Prisoner(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.strategy = random.choice(['C', 'D'])
        self.payoff = 0

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        for neighbor in neighbors:
            my_payoff, neighbor_payoff = self.play_prisoners_dilemma(neighbor)
            self.payoff += my_payoff
            neighbor.payoff += neighbor_payoff

    def play_prisoners_dilemma(self, opponent):
        if self.strategy == 'C' and opponent.strategy == 'C':
            return R, R
        elif self.strategy == 'C' and opponent.strategy == 'D':
            return S, T
        elif self.strategy == 'D' and opponent.strategy == 'C':
            return T, S
        else:
            return P, P

    def advance(self):
        if self.payoff >= 3:
            self.strategy = 'C'
        else:
            self.strategy = 'D'
        self.payoff = 0

class PrisonersDilemmaModel(Model):
    def __init__(self, width, height, N):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            a = Prisoner(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
        self.datacollector = DataCollector(
            {
                "Cooperators": lambda m: self.count_type(m, 'C'),
                "Defectors": lambda m: self.count_type(m, 'D')
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    @staticmethod
    def count_type(model, strategy):
        count = 0
        for agent in model.schedule.agents:
            if agent.strategy == strategy:
                count += 1
        return count

model = PrisonersDilemmaModel(10, 10, 50)

for i in range(100):
    model.step()

data = model.datacollector.get_model_vars_dataframe()

print(data)

plt.figure(figsize=(10, 6))
plt.plot(data["Cooperators"], label="Cooperators (C)", color="green")
plt.plot(data["Defectors"], label="Defectors (D)", color="red")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.title("Prisoner's Dilemma: Cooperators vs. Defectors")
plt.legend()
plt.grid(True)
plt.show()

