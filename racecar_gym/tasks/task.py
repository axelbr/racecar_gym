from abc import abstractmethod, ABC


class Task(ABC):

    @abstractmethod
    def reward(self, agent_id, state, action) -> float:
        pass

    @abstractmethod
    def done(self, agent_id, state) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass
