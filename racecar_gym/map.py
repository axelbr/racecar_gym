from pybullet_utils.bullet_client import BulletClient


class Map:
    def __init__(self, client: BulletClient, model: str):
        self._model = model
        self._client = client
        self._id = self._load_map()

    def _load_map(self) -> int:
        return self._client.loadSDF(self._model, globalScaling=1)


    def reset(self):
        self._id = self._load_map()
