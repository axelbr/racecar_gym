from dependency_injector import containers, providers
from racecar_gym.entities import Vehicle, World
from racecar_gym.racing.agent import Agent
from racecar_gym.racing.specs import VehicleSpec, WorldSpec, TaskSpec, AgentSpec
from racecar_gym.racing.tasks import Task, task_from_spec

from agents.gap_follower import GapFollower
from racecar_gym.bullet.providers import BulletContainer


class Container(containers.DeclarativeContainer):
    vehicle_factory = providers.AbstractFactory(Vehicle)
    world_factory = providers.AbstractFactory(World)
    model_factory = providers.FactoryAggregate(
        vehicle=vehicle_factory,
        world=world_factory
    )

    task_factory = providers.AbstractFactory(Task)


class AgentContainer(containers.DeclarativeContainer):
    vehicle_factory = providers.AbstractFactory(Vehicle)
    world_factory = providers.AbstractFactory(World)

    task_factory = providers.AbstractFactory(Task)
    model_factory = providers.FactoryAggregate(
        vehicle=vehicle_factory,
        world=world_factory
    )
    spec = providers.Object(AgentSpec())
    agent_factory = providers.Factory(Agent,
                                      id=1,
                                      vehicle=providers.Callable(vehicle_factory, spec=spec().vehicle),
                                      task=providers.Callable(vehicle_factory, spec=spec().vehicle)
                                      )


if __name__ == '__main__':
    engine = 'bullet'
    container = AgentContainer()
    if engine == 'bullet':
        container.world_factory.override(BulletContainer.world_factory)
        container.vehicle_factory.override(BulletContainer.vehicle_factory)
        container.task_factory.override(providers.Factory(task_from_spec, ))

    world = container.model_factory('world', spec=WorldSpec(
        name='berlin'))  # container.world_factory(spec=WorldSpec(name='berlin'))
    vehicle1 = container.model_factory('vehicle', spec=VehicleSpec(name='racecar', sensors=[
        'hokuyo']))  # container.vehicle_factory(spec=VehicleSpec(name='racecar', sensors=['hokuyo', 'zed']))
    vehicle2 = container.model_factory('vehicle', spec=VehicleSpec(name='racecar', sensors=[
        'hokuyo']))  # container.vehicle_factory(spec=VehicleSpec(name='racecar', sensors=['hokuyo']))

    task1 = container.task_factory(TaskSpec('time_based', params={'max_time': 120.0, 'time_step': 0.01, 'laps': 2}))
    task2 = container.task_factory(TaskSpec('time_based', params={'max_time': 120.0, 'time_step': 0.01, 'laps': 2}))

    vehicle_spec = VehicleSpec(name='racecar', sensors=['hokuyo'])
    task_spec = TaskSpec('time_based', params={'max_time': 120.0, 'time_step': 0.01, 'laps': 2})
    container.spec.override(AgentSpec(vehicle=vehicle_spec, task=task_spec))
    id = 2
    agent = container.agent_factory
    agent = GapFollower()
    world.init()
    vehicle1.reset(pose=world.initial_pose(position=0))
    vehicle2.reset(pose=world.initial_pose(position=1))
    obs1 = vehicle1.observe()
    obs2 = vehicle2.observe()
    while True:
        if 'hokuyo' in obs1:
            control = agent.action({'lidar': obs1['hokuyo']})
            vehicle1.control(commands={'motor': (control[0], control[1]), 'steering': control[2]})

        if 'hokuyo' in obs2:
            control = agent.action({'lidar': obs2['hokuyo']})
            vehicle2.control(commands={'motor': (control[0], control[1]), 'steering': control[2]})

        obs1 = vehicle1.observe()
        obs2 = vehicle2.observe()
        world.update()
