# simulation

from environment import SAGAscapeModel


def run_simulation(width=50, height=50, num_communities=5, steps=10):
    model = SAGAscapeModel(width=width, height=height, num_communities=num_communities)
    for i in range(steps):
        model.step()
        print(f"TimeStep {i} completed")
    return model